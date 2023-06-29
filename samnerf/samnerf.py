from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import open_clip
import torch
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

from samnerf.encoders.image_encoder import BaseImageEncoder
from samnerf.samnerf_field import SAMNERFField
from samnerf.samnerf_fieldheadnames import SAMNERFFieldHeadNames
from samnerf.samnerf_renderers import CLIPRenderer, MeanRenderer
from samnerf.helper import *

@dataclass
class SAMNERFModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: SAMNERFModel)
    clip_loss_weight: float = 0.1
    
    contrastive_sample_n = 1 # size of constrastive sample points
    contrastive_temperature = 0.5 
    contrastive_loss_weight: float = 0.1
    postive_threshold: float = 0.5 # threshold to distinguish positive and negative for contrastive learning
    
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    num_samnerf_samples: int = 24
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)
    
    
    


class SAMNERFModel(NerfactoModel):
    config: SAMNERFModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.renderer_clip = CLIPRenderer()
        self.renderer_mean = MeanRenderer()

        self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]
        self.samnerf_field = SAMNERFField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            clip_n_dims=self.image_encoder.embedding_dim,
        )


    def get_max_across(self, ray_samples, weights, hashgrid_field, scales_shape, preset_scales=None):
        # TODO smoothen this out
        if preset_scales is not None:
            assert len(preset_scales) == len(self.image_encoder.positives)
            scales_list = torch.tensor(preset_scales)
        else:
            scales_list = torch.linspace(0.0, self.config.max_scale, self.config.n_scales)

        # probably not a good idea bc it's prob going to be a lot of memory
        n_phrases = len(self.image_encoder.positives)
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]
        for i, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                clip_output = self.samnerf_field.get_output_from_hashgrid(
                    ray_samples,
                    hashgrid_field,
                    torch.full(scales_shape, scale, device=weights.device, dtype=hashgrid_field.dtype),
                )
            clip_output = self.renderer_clip(embeds=clip_output, weights=weights.detach())

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
        samnerf_weights, best_ids = torch.topk(weights, self.config.num_samnerf_samples, dim=-2, sorted=False)

        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        samnerf_samples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

        if self.training:
            clip_scales = ray_bundle.metadata["clip_scales"]
            clip_scales = clip_scales[..., None]
            dist = samnerf_samples.spacing_to_euclidean_fn(samnerf_samples.spacing_starts.squeeze(-1)).unsqueeze(-1)
            clip_scales = clip_scales * ray_bundle.metadata["width"] * (1 / ray_bundle.metadata["fx"]) * dist
        else:
            clip_scales = torch.ones_like(samnerf_samples.spacing_starts, device=self.device)

        override_scales = (
            None if "override_scales" not in ray_bundle.metadata else ray_bundle.metadata["override_scales"]
        )
        weights_list.append(weights)
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        samnerf_field_outputs = self.samnerf_field.get_outputs(samnerf_samples, clip_scales)

        if self.training:
            outputs["clip"] = self.renderer_clip(
                embeds=samnerf_field_outputs[SAMNERFFieldHeadNames.CLIP], weights=samnerf_weights.detach()
            )
            outputs["dino"] = self.renderer_mean(
                embeds=samnerf_field_outputs[SAMNERFFieldHeadNames.DINO], weights=samnerf_weights.detach()
            )

        if not self.training:
            with torch.no_grad():
                max_across, best_scales = self.get_max_across(
                    samnerf_samples,
                    samnerf_weights,
                    samnerf_field_outputs[SAMNERFFieldHeadNames.HASHGRID],
                    clip_scales.shape,
                    preset_scales=override_scales,
                )
                outputs["raw_relevancy"] = max_across  # N x B x 1
                outputs["best_scales"] = best_scales.to(self.device)  # N

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        SAMNERF overrides this from base_model since we need to compute the max_across relevancy in multiple batches,
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
            ray_bundle.metadata["override_scales"] = best_scales
            outputs = self.forward(ray_bundle=ray_bundle)
            # standard nerfstudio concatting
            for output_name, output in outputs.items():  # type: ignore
                if output_name == "best_scales":
                    continue
                if output_name == "raw_relevancy":
                    for r_id in range(output.shape[0]):
                        outputs_lists[f"relevancy_{r_id}"].append(output[r_id, ...])
                else:
                    outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        for i in range(len(self.image_encoder.positives)):
            p_i = torch.clip(outputs[f"relevancy_{i}"] - 0.5, 0, 1)
            outputs[f"composited_{i}"] = apply_colormap(p_i / (p_i.max() + 1e-6), ColormapOptions("turbo"))
            mask = (outputs["relevancy_0"] < 0.5).squeeze()
            outputs[f"composited_{i}"][mask, :] = outputs["rgb"][mask, :]
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
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            unreduced_clip = self.config.clip_loss_weight * torch.nn.functional.huber_loss(
                outputs["clip"], batch["clip"], delta=1.25, reduction="none"
            )
            loss_dict["clip_loss"] = unreduced_clip.sum(dim=-1).nanmean()
            unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
            loss_dict["dino_loss"] = unreduced_dino.sum(dim=-1).nanmean()

            if batch['use_contrastive_loss']:
                img_ray_0 = torch.where(batch['indices'][:, 0] == batch['image_idx'][0].to(batch['indices'].device))
                img_ray_1 = torch.where(batch['indices'][:, 0] == batch['image_idx'][1].to(batch['indices'].device))
                img_ray_0 = outputs["clip"][img_ray_0]
                img_ray_1 = outputs["clip"][img_ray_1]
                
                support_ray = torch.randint(0, img_ray_0.size(0), (self.config.contrastive_sample_n,))
                support_ray = img_ray_0[support_ray].detach()
                
                self_support_mask = self.self_support(img_ray_1, support_ray)
                loss_dict['contrastive_loss'] = self.contrastive_loss(self_support_mask, img_ray_0, support_ray)
                
                
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["samnerf"] = list(self.samnerf_field.parameters())
        return param_groups

    def contrastive_loss(self, mask, features, support_feature):
        """
        Args:
            mask: [N, 1]
            features: [N, K]
            support_feature: [1, K]
        """
        labels = mask> self.config.postive_threshold
        
        
        similarity = features @ support_feature.T
        print_shape(similarity)
        exit()
        # contrastive_loss = pos_exp.mean() + max(self.config.epsilon - neg_exp).mean()
        

        
        
        return contrastive_loss
    
    
    def self_support(self, feature_q, support):
        """
        Args:
            feature:[N, K]
            support:[1, K]
        """
        
        print_shape(feature_q)
        print_shape(support)
        
        pred_1 = F.cosine_similarity(feature_q, support, dim=1) # 


        fg_thres = 0.7 #0.9 #0.6
        bg_thres = 0.6 #0.6
        cur_feat = feature_q
        f_h, f_w = feature_q.shape
        
        
        if (pred_1 > fg_thres).sum() > 0:
            fg_feat = cur_feat[(pred_1>fg_thres)] #.mean(-1)
        else:
            fg_feat = cur_feat[torch.topk(pred_1, 12).indices] #.mean(-1)
        if ((1 - pred_1) > bg_thres).sum() > 0:
            bg_feat = cur_feat[((1 - pred_1)>bg_thres)] #.mean(-1)
        else:
            bg_feat = cur_feat[torch.topk((1 - pred_1), 12).indices] #.mean(-1)
                # global proto
                
                
        fg_proto = fg_feat.mean(-1)
        bg_proto = bg_feat.mean(-1)

        print_shape(fg_feat)
        print(torch.norm(fg_feat))
        exit()
        # local proto
        fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True) # 1024, N1
        bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True) # 1024, N2
        cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True) # 1024, N3

        cur_feat_norm_t = cur_feat_norm.t() # N3, 1024
        fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0 # N3, N1
        bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0 # N3, N2

        fg_sim = fg_sim.softmax(-1)
        bg_sim = bg_sim.softmax(-1)

        fg_proto_local = torch.matmul(fg_sim, fg_feat.t()) # N3, 1024
        bg_proto_local = torch.matmul(bg_sim, bg_feat.t()) # N3, 1024

        fg_proto_local = fg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0) # 1024, N3
        bg_proto_local = bg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0) # 1024, N3



        # SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self_support_loss(img_ray_1, support_ray)
        self_support = support * 0.5 + fg_proto * 0.5
        
        self_support_mask = F.cosine_similarity(feature_q, self_support)
                    
        return self_support_mask