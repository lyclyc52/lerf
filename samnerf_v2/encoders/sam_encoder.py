from dataclasses import dataclass, field
from typing import Tuple, Type

import torch
import torchvision
import numpy as np


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
    model_type: str = "default"
    sam_n_dims: int = 256
    sam_checkpoint: str = "/ssddata/yliugu/lerf/dependencies/Grounded-Segment-Anything/checkpoint/sam_vit_h_4b8939.pth"



class SAMNetwork(BaseImageEncoder):
    def __init__(self, config: SAMNetworkConfig):
        super().__init__()
        self.config = config

        sam = sam_model_registry[self.config.model_type](checkpoint=self.config.sam_checkpoint)
        sam.to(device="cuda")
        self.model = SamPredictor(sam)
        
        self.positive_input = ViewerText("LERF Positives", "", cb_hook=self.gui_cb)

        self.positives = self.set_positives(self.positive_input.value)


    @property
    def name(self) -> str:
        return "sam_{}".format(self.config.model_type)

    @property
    def embedding_dim(self) -> int:
        return self.config.sam_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

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
    
    def decode_feature(self, feature, image, index):
        
        if not self.model.is_image_set:
            self.model.set_image(image)
        feature = feature.permute(2,0,1)
        self.model.set_torch_feature(feature)
        positive = np.array([self.positives[index]])
        masks, scores, logits = self.model.predict(
            point_coords=positive,
            point_labels=np.array([1]),
            multimask_output=True,
        )
        return masks, scores, logits
            
