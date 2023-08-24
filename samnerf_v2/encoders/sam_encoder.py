from dataclasses import dataclass, field
from typing import Tuple, Type

import torch
import torchvision
import numpy as np
import cv2
import torch.nn.functional as F


try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    assert False, "segment_anything is not installed"

from samnerf_v2.encoders.image_encoder import (BaseImageEncoder,
                                         BaseImageEncoderConfig)
from nerfstudio.viewer.server.viewer_elements import ViewerText


@dataclass
class SAMNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: SAMNetwork)
    model_type: str = "vit_l"
    sam_n_dims: int = 256
    sam_checkpoint: str = "/ssddata/yliugu/lerf/dependencies/sam-hq/pretrained_checkpoint/sam_hq_vit_l.pth"
    # sam_checkpoint: str = '/ssddata/yliugu/lerf/dependencies/Grounded-Segment-Anything/checkpoint/sam_vit_h_4b8939.pth'



class SAMNetwork(BaseImageEncoder):
    def __init__(self, config: SAMNetworkConfig):
        super().__init__()
        self.config = config

        sam = sam_model_registry[self.config.model_type](checkpoint=self.config.sam_checkpoint)
        sam.to(device="cuda")
        self.model = SamPredictor(sam)
        
        self.positive_input = ViewerText("SAM Positives", "", cb_hook=self.gui_cb)

        self.positives = self.set_positives(self.positive_input.value)


    @property
    def name(self) -> str:
        return "sam_{}".format(self.config.model_type)

    @property
    def embedding_dim(self) -> int:
        return self.config.sam_n_dims
    
    def gui_cb(self,element):
        self.positives = self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        positives = []
        for p in text_list:
            p = p.split(',')
            coord = [int(j) for j in p]
            positives.append(np.array(coord))
        return positives

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]

    def encode_image(self, input):
        self.model.set_image(input)
        return self.model.features
    
    def set_model_image_info(self, image):
        input_image = self.model.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.model.reset_image()

        self.model.original_size = image.shape[:2]
        self.model.input_size = tuple(transformed_image.shape[-2:])
        self.model.is_image_set = True
        
    def set_torch_feature(
        self,
        features: np.ndarray,
    ) -> None:
      target_feature_size = self.model.model.image_encoder.img_size // self.model.model.image_encoder.patch_size
      if features.shape[1] != target_feature_size or features.shape[2] != target_feature_size:
        f_h, f_w = features.shape[1:]
        max_length = max(f_h, f_w)
        h,w = int(np.floor(target_feature_size * f_h / max_length)), int(np.floor(target_feature_size * f_w / max_length))
        features = cv2.resize(features.transpose(1,2,0), (w,h), interpolation = cv2.INTER_NEAREST )
        features = features.transpose(2,0,1)
        features = torch.from_numpy(features[None,...]).to(self.model.device)
        
        padh = target_feature_size - h
        padw = target_feature_size - w
        self.features = F.pad(features, (0, padw, 0, padh))
      else:
        self.features = torch.from_numpy(features[None,...]).to(self.model.device)

    
    def decode_feature(self, feature, image, position):
        image = (image.cpu().numpy() * 255).astype(np.uint8)
        

        self.set_model_image_info(image)
        feature = feature.permute(2,0,1).cpu().numpy()
        self.set_torch_feature(feature)
        masks, scores, logits = self.model.predict(
            point_coords=position,
            point_labels=np.array([1]),
            multimask_output=True,
        )
        return masks, scores, logits
            
