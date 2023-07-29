import json
import os
from pathlib import Path

import numpy as np
import torch
from lerf.data.utils.feature_dataloader import FeatureDataloader
from lerf.data.utils.patch_embedding_dataloader import PatchEmbeddingDataloader
from lerf.encoders.image_encoder import BaseImageEncoder
from tqdm import tqdm
import cv2

class SAMDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: BaseImageEncoder,
        image_list: torch.Tensor = None,
        cache_path: str = None,
    ):
        assert "tile_size_range" in cfg
        assert "tile_size_res" in cfg
        assert "stride_scaler" in cfg
        assert "image_shape" in cfg
        assert "model_name" in cfg

        self.tile_sizes = torch.linspace(*cfg["tile_size_range"], cfg["tile_size_res"]).to(device)
        self.strider_scaler_list = [self._stride_scaler(tr.item(), cfg["stride_scaler"]) for tr in self.tile_sizes]

        self.feature_size = 64
        self.feature_scale = np.max(np.array(cfg["image_shape"])) / self.feature_size


        self.model = model
        self.embed_size = self.model.embedding_dim
        self.data_dict = {}
        self.feature_list = []
        super().__init__(cfg, device, image_list, cache_path)

    def __call__(self, img_points, get_mask=False):
        return self._random_scales(img_points, get_mask)


    def _stride_scaler(self, tile_ratio, stride_scaler):
        return np.interp(tile_ratio, [0.05, 0.15], [1.0, stride_scaler])

    def load(self):
        # don't create anything, PatchEmbeddingDataloader will create itself
        cache_info_path = self.cache_path.with_suffix(".info")

        # check if cache exists
        if not cache_info_path.exists():
            raise FileNotFoundError

        # if config is different, remove all cached content
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            for f in os.listdir(self.cache_path):
                os.remove(os.path.join(self.cache_path, f))
            raise ValueError("Config mismatch")
        
        self.feature_list = torch.from_numpy(np.load(self.cache_path.with_suffix(".npy"))).to(self.device)


    def create(self, image_list):
        for img in image_list:
            img = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            feature = self.model.encode_image(img)
            self.feature_list.append(feature)
            

        self.feature_list = torch.cat(self.feature_list).to(self.device)

    def save(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))
            
        np.save(self.cache_path, self.feature_list.cpu())
        

    def _random_scales(self, img_points, get_mask=False):
        # img_points: (B, 3) 
        img_points = img_points.to(self.device)
        img_ind = img_points[:, 0]
        feature_coord = img_points[:, 1:] / self.feature_scale

        
        x_ind = torch.floor(feature_coord[:, 0]).long()
        y_ind = torch.floor(feature_coord[:, 1]).long()
        x_left, x_right = x_ind, torch.where(x_ind + 1 < self.feature_size - 1, x_ind + 1, self.feature_size - 1)
        y_top, y_bot = y_ind, torch.where(y_ind + 1 < self.feature_size - 1, y_ind + 1, self.feature_size - 1)

        topleft = self.feature_list[img_ind, :, x_left, y_top].to(self.device)
        topright = self.feature_list[img_ind, :, x_right, y_top].to(self.device)
        botleft = self.feature_list[img_ind, :,  x_left, y_bot].to(self.device)
        botright = self.feature_list[img_ind, :, x_right, y_bot].to(self.device)
        
        
        right_w = (feature_coord[:, 0] - x_ind).to(self.device)  
        top = torch.lerp(topleft, topright, right_w[:, None])
        bot = torch.lerp(botleft, botright, right_w[:, None])

        bot_w = (feature_coord[:, 1] - y_ind).to(self.device)  
        feature = torch.lerp(top, bot, bot_w[:, None])
        
        mask = None
        if get_mask:
            mask = self.mask[img_points[:, 1], img_points[:, 2]]
            
        return feature, mask

    def generate_mask(self, image_batch, position):
        feature = self.feature_list[position[0][0]]
        position = position[0][1:].numpy()
        position = np.array([[position[1], position[0]]])

        masks, scores, logits = self.model.decode_feature(feature.permute(1,2,0), 
                                                        image_batch['image'][0], 
                                                        position)

        
        self.mask = torch.from_numpy(masks[np.argmax(scores)]).to(self.device)