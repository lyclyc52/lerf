import typing

import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from samnerf.data.utils.dino_extractor import ViTExtractor
from samnerf.data.utils.feature_dataloader import FeatureDataloader
from tqdm import tqdm


class SAMDataloader(FeatureDataloader):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        cache_path: str = None,
    ):
        assert "image_shape" in cfg
        super().__init__(cfg, device, image_list, cache_path)

    def create(self, image_list):
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        
        sam.to(device=self.device)

        mask_generator = SamAutomaticMaskGenerator(sam)
        
        dino_embeds = []
        for image in tqdm(preproc_image_lst, desc="SAM", total=len(image_list), leave=False):
            with torch.no_grad():
                masks = mask_generator.generate(image)
            dino_embeds.append(masks.cpu().detach())

        self.data = torch.stack(dino_embeds, dim=0)

    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)
