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

class HQSAMDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: BaseImageEncoder,
        image_list: torch.Tensor = None,
        cache_path: str = None,
    ):
        assert "image_shape" in cfg
        assert "model_name" in cfg

        self.feature_size = 64
        self.feature_scale = np.max(np.array(cfg["image_shape"])) / self.feature_size

        # If supersampling, generate image-sized feature maps
        self.supersampling = cfg.get("supersampling", False)
        self.original_feature_size = 64     # original SAM feature size

        if self.supersampling:
            img_max_dim = np.max(np.array(cfg["image_shape"]))
            target_feature_size = 64 * cfg["supersampling_factor"]
            self.feature_size = min(img_max_dim, target_feature_size)
            self.feature_scale = img_max_dim / self.feature_size

        self.model = model
        self.embed_size = self.model.embedding_dim
        self.data_dict = {}
        self.feature_list = []
        self.interm_feature_list = []
        for i in range(4):
            self.interm_feature_list.append([])
        super().__init__(cfg, device, image_list, cache_path)

    def __call__(self, img_points, get_mask=False):
        return self._random_scales(img_points, get_mask)


    def _stride_scaler(self, tile_ratio, stride_scaler):
        return np.interp(tile_ratio, [0.05, 0.15], [1.0, stride_scaler])

    def load(self):
        # don't create anything, PatchEmbeddingDataloader will create itself
        cache_info_path = self.cache_path.with_suffix(".info")
        cache_npy_path_0 = self.cache_path.with_suffix(".npy")
        cache_npy_path_1 = Path(str(self.cache_path) + ("_hq0.npy"))

        # check if cache exists
        if not cache_info_path.exists() or not cache_npy_path_0.exists() or not cache_npy_path_1.exists():
            raise FileNotFoundError

        # if config is different, remove all cached content
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            for f in os.listdir(self.cache_path):
                os.remove(os.path.join(self.cache_path, f))
            raise ValueError("Config mismatch")
        
        device = 'cpu' if self.supersampling else self.device
        self.feature_list = torch.from_numpy(np.load(self.cache_path.with_suffix(".npy"))).to('cpu')
        for i in range(4):
            self.interm_feature_list[i] = torch.from_numpy(np.load(str(self.cache_path) + (f"_hq{i}.npy"))).to('cpu')


    def get_supersampled_feature(self, img):
        num_samples = int(np.ceil(self.feature_size / self.original_feature_size))
        full_feature = None

        for i in range(num_samples):
            for j in range(num_samples):
                x = num_samples // 2 - i - 1
                y = num_samples // 2 - j - 1
                
                # print(num_samples)
                # x = num_samples // 2 - 200 - 1
                # y = num_samples // 2 - 200 - 1
                
                T = np.array([[1, 0, x], [0, 1, y]]).astype(np.float32)

                img_shifted = cv2.warpAffine(img, T, (img.shape[1], img.shape[0]))
                cv2.imwrite(f'save_{i}_{j}.png', img_shifted)
                feature, interm_feature = self.model.encode_image(img_shifted)
                if full_feature is None:
                    full_feature = torch.zeros((*feature.shape[:2], self.feature_size, self.feature_size), 
                                               device=self.device, dtype=feature.dtype)
                x_inds = np.arange(i, self.feature_size, num_samples)
                y_inds = np.arange(j, self.feature_size, num_samples)
                full_feature[:, :, y_inds[:, None], x_inds] = feature[:, :, :len(y_inds), :len(x_inds)]
        return full_feature

    def create(self, image_list):
        for img in tqdm(image_list):
            img = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            if self.supersampling:
                feature = self.get_supersampled_feature(img)
            else:
                feature, interm_feature = self.model.encode_image(img)
            # self.feature_list.append(feature.to('cpu' if self.supersampling else self.device))
            self.feature_list.append(feature.to('cpu'))
            for l in range(len(interm_feature)):
                self.interm_feature_list[l].append(interm_feature[l].permute(0,3,1,2).to('cpu'))
            
        # self.feature_list = torch.cat(self.feature_list).to('cpu' if self.supersampling else self.device)
        self.feature_list = torch.cat(self.feature_list).to('cpu')
        for l in range(len(self.interm_feature_list)):
            # self.interm_feature_list[l] = torch.cat(self.interm_feature_list[l]).to('cpu' if self.supersampling else self.device)
            self.interm_feature_list[l] = torch.cat(self.interm_feature_list[l]).to('cpu')
            

    def save(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))
            
        np.save(self.cache_path, self.feature_list.cpu())
        for i in range(len(self.interm_feature_list)):
            np.save(str(self.cache_path) + (f'_hq{i}.npy'), self.interm_feature_list[i].cpu())
        

    def _random_scales(self, img_points, get_mask=False):
        feautre_device = self.feature_list.device

        # img_points: (B, 3) 
        img_points = img_points.to(feautre_device)
        img_ind = img_points[:, 0]
        
        feature_coord = img_points[:, 1:] / self.feature_scale
        
        x_ind = torch.floor(feature_coord[:, 0]).long()
        y_ind = torch.floor(feature_coord[:, 1]).long()
        x_left, x_right = x_ind, torch.where(x_ind + 1 < self.feature_size - 1, x_ind + 1, self.feature_size - 1)
        y_top, y_bot = y_ind, torch.where(y_ind + 1 < self.feature_size - 1, y_ind + 1, self.feature_size - 1)


        def sample_feature(feature_map):
            topleft = feature_map[img_ind, :, x_left, y_top]
            topright = feature_map[img_ind, :, x_right, y_top]
            botleft = feature_map[img_ind, :,  x_left, y_bot]
            botright = feature_map[img_ind, :, x_right, y_bot]
            
            
            right_w = (feature_coord[:, 0] - x_ind)
            top = torch.lerp(topleft, topright, right_w[:, None])
            bot = torch.lerp(botleft, botright, right_w[:, None])

            bot_w = (feature_coord[:, 1] - y_ind)
            feature = torch.lerp(top, bot, bot_w[:, None]).to(self.device)
            return feature
        
        feature = sample_feature(self.feature_list)
        
        interm_feature = []
        for l in range(len(self.interm_feature_list)):
            interm_feature.append(sample_feature(self.interm_feature_list[l]))
        
        mask = None
        if get_mask:
            img_points = img_points.to(self.mask.device)
            mask = self.mask[img_points[:, 1], img_points[:, 2]]
            
        return [feature, interm_feature], mask

    def generate_mask(self, image_batch, position):
        feature = self.feature_list[position[0][0]]
        position = position[0][1:].numpy()
        position = np.array([[position[1], position[0]]])

        masks, scores, logits = self.model.decode_feature(feature.permute(1,2,0), 
                                                        image_batch['image'][0], 
                                                        position)

        self.mask = torch.from_numpy(masks[np.argmax(scores)]).to(self.device)