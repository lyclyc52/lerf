from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
from nerfstudio.viewer.server.viewer_elements import *
from torch.nn import Parameter

from samnerf_v2.encoders.image_encoder import BaseImageEncoder
from samnerf_v2.samnerf_field import LERFField
from samnerf_v2.lerf_fieldheadnames import LERFFieldHeadNames
from samnerf_v2.lerf_renderers import CLIPRenderer, MeanRenderer
from samnerf_v2.helper import print_shape
import imageio
from typing import Literal
import os


@dataclass
class LERFModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: LERFModel)
    feature_loss_weight: float = 0.1
    n_scales: int = 30
    max_scale: float = 1.5
    
    feature_type: Literal['clip', 'xdecoder', 'sam', 'hqsam'] = 'sam' 
    
    use_contrastive: bool = False
    contrastive_sample_n = 1 # size of constrastive sample points
    contrastive_temperature = 20
    contrastive_loss_weight: float = 0.1
    postive_threshold: float = 0.6 # threshold to distinguish positive and negative for contrastive learning
    
    """maximum scale used to compute relevancy with"""
    num_lerf_samples: int = 24
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)


class LERFModel(NerfactoModel):
    config: LERFModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.renderer_feature = CLIPRenderer() if self.config.feature_type == 'clip' else MeanRenderer()
        self.renderer_mean = MeanRenderer()
        self.renderer_contrastive = MeanRenderer()

        self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]
        self.lerf_field = LERFField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            clip_n_dims=self.image_encoder.embedding_dim,
            feature_type=self.config.feature_type,
            use_contrastive=self.config.use_contrastive
        )
        
        self.latest_featuremap = None
        self.latest_constrastive_featuremap = None
        self.latest_hq_featuremap = None
        self.latest_rgb = None
        self.feautre_save_path = ViewerText(name="Feature save path",
                                         default_value="./debug/featuremap.npy")
        self.save_feature_button = ViewerButton(name="Save feature", 
                                             cb_hook=self.save_featuremap)

        # populate some viewer logic
        # TODO use the values from this code to select the scale
        
        # def scale_cb(element):
        #     self.config.n_scales = element.value

        # self.n_scale_slider = ViewerSlider("N Scales", 15, 5, 30, 1, cb_hook=scale_cb)

        # def max_cb(element):
        #     self.config.max_scale = element.value

        # self.max_scale_slider = ViewerSlider("Max Scale", 1.5, 0, 5, 0.05, cb_hook=max_cb)

        # def hardcode_scale_cb(element):
        #     self.hardcoded_scale = element.value

        # self.hardcoded_scale_slider = ViewerSlider(
        #     "Hardcoded Scale", 1.0, 0, 5, 0.05, cb_hook=hardcode_scale_cb, disabled=True
        # )

        # def single_scale_cb(element):
        #     self.n_scale_slider.set_disabled(element.value)
        #     self.max_scale_slider.set_disabled(element.value)
        #     self.hardcoded_scale_slider.set_disabled(not element.value)

        # self.single_scale_box = ViewerCheckbox("Single Scale", False, cb_hook=single_scale_cb)
        
    def save_featuremap(self, button):
        save_dir = os.path.dirname(self.feautre_save_path.value)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.latest_featuremap is not None:
            np.save(self.feautre_save_path.value, self.latest_featuremap)
            
        if self.latest_hq_featuremap is not None:
            for i in range(len(self.latest_hq_featuremap)):
                np.save(self.feautre_save_path.value.replace('.npy', f"_hq{i}.npy"), self.latest_hq_featuremap[i])
        
        if self.latest_constrastive_featuremap is not None:
            contrastive_path = self.feautre_save_path.value.replace(".npy", "_ctr.npy")
            np.save(contrastive_path, self.latest_constrastive_featuremap)
        
        if self.latest_rgb is not None:
            rgb_path = self.feautre_save_path.value.replace(".npy", ".png")

            rgb = self.latest_rgb * 255
            rgb = rgb.astype(np.uint8)

            imageio.imwrite(rgb_path, rgb)
            
        print(f"Saved featuremap to {self.feautre_save_path.value}")

    def get_max_across(self, ray_samples, weights, hashgrid_field, scales_shape, preset_scales=None):
        # TODO smoothen this out
        if preset_scales is not None:
            assert len(preset_scales) == len(self.image_encoder.positives)
            scales_list = preset_scales.clone().detach()
        else:
            scales_list = torch.linspace(0.0, self.config.max_scale, self.config.n_scales)

        # probably not a good idea bc it's prob going to be a lot of memory
        n_phrases = len(self.image_encoder.positives)
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]
        for i, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                clip_output = self.lerf_field.get_output_from_hashgrid(
                    ray_samples,
                    hashgrid_field,
                    torch.full(scales_shape, scale, device=weights.device, dtype=hashgrid_field.dtype),
                )
            clip_output = self.renderer_feature(embeds=clip_output, weights=weights.detach())

            for j in range(n_phrases):
                if preset_scales is None or j == i:
                    probs = self.image_encoder.get_relevancy(clip_output, j)
                    pos_prob = probs[..., 0:1]
                    if n_phrases_maxs[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                        n_phrases_maxs[j] = scale
                        n_phrases_sims[j] = pos_prob
        return torch.stack(n_phrases_sims), torch.Tensor(n_phrases_maxs)

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)

        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples)

        lerf_weights, best_ids = torch.topk(weights, self.config.num_lerf_samples, dim=-2, sorted=False)

        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        lerf_samples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

        if self.training:
            if self.config.feature_type == 'clip':
                clip_scales = ray_bundle.metadata["clip_scales"]
                clip_scales = clip_scales[..., None]
                dist = lerf_samples.spacing_to_euclidean_fn(lerf_samples.spacing_starts.squeeze(-1)).unsqueeze(-1)
                clip_scales = clip_scales * ray_bundle.metadata["width"] * (1 / ray_bundle.metadata["fx"]) * dist
        else:
            if self.config.feature_type == 'clip':
                clip_scales = torch.ones_like(lerf_samples.spacing_starts, device=self.device)


        override_scales = (
            None if "override_scales" not in ray_bundle.metadata else ray_bundle.metadata["override_scales"]
        )
        weights_list.append(weights)
        
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        
        if self.config.feature_type == 'clip':
            lerf_field_outputs = self.lerf_field.get_outputs(lerf_samples, clip_scales)
        elif self.config.feature_type == 'sam' or self.config.feature_type == 'hqsam':
            lerf_field_outputs = self.lerf_field.get_outputs(lerf_samples)

        if self.training:
            outputs["feature"] = self.renderer_feature(
                embeds=lerf_field_outputs[LERFFieldHeadNames.FEATURE], weights=lerf_weights.detach()
            )
            outputs["dino"] = self.renderer_mean(
                embeds=lerf_field_outputs[LERFFieldHeadNames.DINO], weights=lerf_weights.detach()
            )
            
            if self.config.use_contrastive:
                outputs["contrastive"] = self.renderer_contrastive(
                    embeds=lerf_field_outputs[LERFFieldHeadNames.CONTRASTIVE], weights=lerf_weights.detach()
                )
            if self.config.feature_type == 'hqsam':
                outputs['hqsam_feature'] = []
                for f in lerf_field_outputs[LERFFieldHeadNames.ADVANCED_FEATURE]:
                    outputs['hqsam_feature'].append(self.renderer_feature(
                        embeds=f, weights=lerf_weights.detach()
                    ))
                
        if not self.training:
            with torch.no_grad():
                if self.config.feature_type == 'clip':
                    max_across, best_scales = self.get_max_across(
                        lerf_samples,
                        lerf_weights,
                        lerf_field_outputs[LERFFieldHeadNames.HASHGRID],
                        clip_scales.shape,
                        preset_scales=override_scales,
                    )
                    outputs["raw_relevancy"] = max_across  # N x B x 1
                    outputs["best_scales"] = best_scales.to(self.device)  # N
                    outputs["feature"] = self.renderer_feature(
                        embeds=lerf_field_outputs[LERFFieldHeadNames.FEATURE], weights=lerf_weights.detach()
                    )
                elif self.config.feature_type == 'sam' or self.config.feature_type == 'hqsam':
                    outputs["feature"] = self.renderer_feature(
                        embeds=lerf_field_outputs[LERFFieldHeadNames.FEATURE], weights=lerf_weights.detach()
                    )
                    if self.config.use_contrastive:
                        outputs["contrastive"] = self.renderer_contrastive(
                            embeds=lerf_field_outputs[LERFFieldHeadNames.CONTRASTIVE], weights=lerf_weights.detach()
                        )
                    if self.config.feature_type == 'hqsam':
                        outputs['hqsam_feature'] = []
                        for f in lerf_field_outputs[LERFFieldHeadNames.ADVANCED_FEATURE]:
                            outputs['hqsam_feature'].append(self.renderer_feature(
                                embeds=f, weights=lerf_weights.detach()
                            ))
                
        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        LERF overrides this from base_model since we need to compute the max_across relevancy in multiple batches,
        which are not independent since they need to use the same scale
        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        # TODO(justin) implement max across behavior
        
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)  # dict from name:list of outputs (1 per bundle)

        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            
            # take the best scale for each query across each ray bundle
            
            if self.config.feature_type == 'clip':
                if i == 0:
                    best_scales = outputs["best_scales"]
                    best_relevancies = [m.max() for m in outputs["raw_relevancy"]]
                else:
                    for phrase_i in range(outputs["best_scales"].shape[0]):
                        m = outputs["raw_relevancy"][phrase_i, ...].max()
                        if m > best_relevancies[phrase_i]:
                            best_scales[phrase_i] = outputs["best_scales"][phrase_i]
                            best_relevancies[phrase_i] = m
        # re-render the max_across outputs using the best scales across all batches
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            if self.config.feature_type == 'clip':
                ray_bundle.metadata["override_scales"] = best_scales
            outputs = self.forward(ray_bundle=ray_bundle)

            # standard nerfstudio concatting
            for output_name, output in outputs.items():  # type: ignore
                if output_name == "best_scales":
                    continue
                elif output_name == "raw_relevancy":
                    for r_id in range(output.shape[0]):
                        outputs_lists[f"relevancy_{r_id}"].append(output[r_id, ...])
                else:
                    outputs_lists[output_name].append(output)
        outputs = {}
        outputs['hqsam_feature'] = [[] for _ in range(4)]
        for output_name, outputs_list in outputs_lists.items():
            if output_name == 'hqsam_feature':
                for l in outputs_list:
                    for i in range(len(l)):
                        outputs[output_name][i].append(l[i])  # type: ignore
                for i in range(len(outputs['hqsam_feature'])):    
                    outputs['hqsam_feature'][i] = torch.cat(outputs['hqsam_feature'][i]).view(image_height, image_width, -1)
            
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            else:
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore

        for i in range(len(self.image_encoder.positives)):
            if self.config.feature_type == 'clip':
                p_i = torch.clip(outputs[f"relevancy_{i}"] - 0.5, 0, 1)
                outputs[f"composited_{i}"] = apply_colormap(p_i / (p_i.max() + 1e-6), ColormapOptions("turbo"))
                mask = (outputs["relevancy_0"] < 0.5).squeeze()
                outputs[f"composited_{i}"][mask, :] = outputs["rgb"][mask, :]
            elif self.config.feature_type == 'sam':
                # print_shape(outputs['rgb'])
                position = np.array(self.image_encoder.positives)
                masks, scores, logits = self.image_encoder.decode_feature(outputs['feature'], outputs['rgb'], position)
                mask = outputs["rgb"]
                mask[masks[0], :] = mask[masks[0], :] * 0.5 + torch.tensor([0.6,0.1,0.1])[None, None, ...].to(mask.device)
                outputs[f"composited_0"] = torch.clip(mask, 0, 1)

                
        if image_width >= 512:
            self.latest_featuremap = outputs['feature'].detach().cpu().numpy()
            if 'contrastive' in outputs:
                self.latest_constrastive_featuremap = outputs['contrastive'].detach().cpu().numpy()
            if "hqsam_feature" in outputs and self.config.feature_type == 'hqsam':
                self.latest_hq_featuremap = []
                for i in range(len(outputs['hqsam_feature'])):
                    self.latest_hq_featuremap.append(outputs['hqsam_feature'][i].detach().cpu().numpy())
                    outputs[f'hqsam_feature_{i}'] = outputs['hqsam_feature'][i]
            
                    
                
            self.latest_rgb = outputs['rgb'].detach().cpu().numpy()
        outputs.pop('hqsam_feature')
        return outputs

    def _get_outputs_nerfacto(self, ray_samples: RaySamples):
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }


        return field_outputs, outputs, weights

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        
        if self.training:
            if batch['train_feature']:
                loss_dict = {}
            else:
                loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
                
            
            if batch['use_contrastive_loss']:
                loss_dict = {}
                support_ray = outputs["contrastive"][batch['sample_idx']]
                loss_dict['contrastive_loss'] =  self.config.contrastive_loss_weight * self.contrastive_loss(batch['mask'], outputs['contrastive'], support_ray) 
            if "feature" in batch:
                loss_dict["feature_loss"] = 0
                if self.config.feature_type == 'hqsam':
                    feature, hq_feature = batch["feature"]
                    for i in range(len(hq_feature)):
                        unreduced_feature= self.config.feature_loss_weight * torch.nn.functional.huber_loss(
                            outputs["hqsam_feature"][i], hq_feature[i], delta=1.25, reduction="none"
                        ) 
                        loss_dict["feature_loss"] += unreduced_feature.sum(dim=-1).nanmean()
                else:
                    feature = batch["feature"]
                unreduced_clip = self.config.feature_loss_weight * torch.nn.functional.huber_loss(
                    outputs["feature"], feature, delta=1.25, reduction="none"
                )
                loss_dict["feature_loss"] += unreduced_clip.sum(dim=-1).nanmean()

            if "dino" in batch:
                unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
                loss_dict["dino_loss"] = unreduced_dino.sum(dim=-1).nanmean()
        else:
            loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
            
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["lerf"] = list(self.lerf_field.parameters())
        return param_groups

    def contrastive_loss(self, mask, features, support_feature):
        """
        Referring to CLIP https://arxiv.org/pdf/2103.00020.pdf
        Args:
            mask: [N, 1]
            features: [N, K]
            support_feature: [1, K]
        """
        labels = (mask> self.config.postive_threshold).to(torch.float32)
        # print(torch.exp(-torch.sum((features - support_feature)**2, -1)).shape)
        # exit()
        # similarity = features @ support_feature.T * torch.exp(torch.tensor(self.config.contrastive_temperature))
        similarity = torch.exp(-torch.tensor(self.config.contrastive_temperature) * torch.sum((features - support_feature)**2, -1))

        
        contrastive_loss = torch.nn.functional.binary_cross_entropy_with_logits(similarity, labels)
        
        return contrastive_loss
    
    def similar_func(self, feature_q, support):
        phrases_embeds = torch.cat([support, self.image_encoder.neg_embeds], dim=0)
        p = phrases_embeds.to(feature_q.dtype)  # phrases x 512
        output = torch.mm(feature_q, p.T)  # rays x phrases
        positive_vals = output[..., :1]  # rays x 1
        negative_vals = output[..., 1:]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.image_encoder.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        score = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.image_encoder.negatives), 2))[
            :, 0, :
        ]
        return score[..., 0]
    
    def self_support(self, feature_q, support):
        """
        Args:
            feature:[N, K]
            support:[1, K]
        """

        # pred_1 = F.cosine_similarity(feature_q, support, dim=1) # 
        # print_shape(pred_1)
        pred_1 = self.similar_func(feature_q, support)

        fg_thres = 0.7 #0.9 #0.6
        bg_thres = 0.6 #0.6
        cur_feat = feature_q
        f_h, f_w = feature_q.shape
        
        
        if (pred_1 > fg_thres).sum() > 0:
            fg_feat = cur_feat[(pred_1>fg_thres)] # [N_f, K]
        else:
            fg_feat = cur_feat[torch.topk(pred_1, 12).indices] 
        if ((1 - pred_1) > bg_thres).sum() > 0:
            bg_feat = cur_feat[((1 - pred_1)>bg_thres)] # [N_b, K]
        else:
            bg_feat = cur_feat[torch.topk((1 - pred_1), 12).indices] 
                # global proto
                
                
        fg_proto = fg_feat.mean(0)
        bg_proto = bg_feat.mean(-1)

        # local proto
        # fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True) # 1024, N1
        # bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True) # 1024, N2
        # cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True) # 1024, N3

        fg_feat_t = fg_feat.t() 
        bg_feat_t = bg_feat.t() 
        
        fg_sim = torch.matmul(cur_feat, fg_feat_t) * 2.0 # N, N_f
        bg_sim = torch.matmul(cur_feat, bg_feat_t) * 2.0 # N, N_b

        fg_sim = fg_sim.softmax(-1)
        bg_sim = bg_sim.softmax(-1)

        fg_proto_local = torch.matmul(fg_sim, fg_feat) # N, K
        bg_proto_local = torch.matmul(bg_sim, bg_feat) # N, K

        # SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self_support_loss(img_ray_1, support_ray)
        self_support = support * 0.5 + fg_proto * 0.5
        
        # self_support_mask = F.cosine_similarity(feature_q, self_support)
        self_support_mask = self.similar_func(feature_q, self_support)

                    
        return self_support_mask