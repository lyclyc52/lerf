# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Datamanager.
"""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import yaml
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper
from rich.progress import Console

CONSOLE = Console(width=120)

from samnerf_v2.data.utils.dino_dataloader import DinoDataloader
from samnerf_v2.data.utils.pyramid_embedding_dataloader import PyramidEmbeddingDataloader
from samnerf_v2.data.utils.xdecoder_dataloader import XDecoderDataloader
from samnerf_v2.data.utils.sam_datalodaer import SAMDataloader
from samnerf_v2.encoders.image_encoder import BaseImageEncoder
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig


@dataclass
class LERFDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: LERFDataManager)
    patch_tile_size_range: Tuple[int, int] = (0.05, 0.5)
    patch_tile_size_res: int = 7
    patch_stride_scaler: float = 0.5
    feature_type: Literal['clip', 'xdecoder', 'sam'] = 'sam' 
    contrastive_starting_epoch: int = 5000
    use_contrastive: bool = False
    supersampling: bool = False
    supersampling_factor: int = 4


class LERFDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: LERFDataManagerConfig

    def __init__(
        self,
        config: LERFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        self.image_encoder: BaseImageEncoder = kwargs["image_encoder"]
        images = [self.train_dataset[i]["image"].permute(2, 0, 1)[None, ...] for i in range(len(self.train_dataset))]
        images = torch.cat(images)

        cache_dir = f"outputs/{self.config.dataparser.data.name}"
        clip_cache_path = Path(osp.join(cache_dir, f"{self.image_encoder.name}"))
        dino_cache_path = Path(osp.join(cache_dir, "dino.npy"))
        # NOTE: cache config is sensitive to list vs. tuple, because it checks for dict equality
        self.dino_dataloader = DinoDataloader(
            image_list=images,
            device=self.device,
            cfg={"image_shape": list(images.shape[2:4])},
            cache_path=dino_cache_path,
        )
        torch.cuda.empty_cache()
        self.feature_type = self.config.feature_type
        
        if self.feature_type == "clip":
            self.feature_interpolator = PyramidEmbeddingDataloader(
                image_list=images,
                device=self.device,
                cfg={
                    "tile_size_range": [0.05, 0.5],
                    "tile_size_res": 7,
                    "stride_scaler": 0.5,
                    "image_shape": list(images.shape[2:4]),
                    "model_name": self.image_encoder.name,
                },
                cache_path=clip_cache_path,
                model=self.image_encoder,
            )
        elif self.feature_type == 'xdecoder':
            self.feature_interpolator = XDecoderDataloader(
                image_list=images,
                device=self.device,
                cfg={
                    "tile_size_range": [0.05, 0.5],
                    "tile_size_res": 7,
                    "stride_scaler": 0.5,
                    "image_shape": list(images.shape[2:4]),
                    "model_name": self.image_encoder.name,
                },
                cache_path=clip_cache_path,
                model=self.image_encoder,
            )
        elif self.feature_type == 'sam':
            self.feature_interpolator = SAMDataloader(
                image_list=images,
                device=self.device,
                cfg={
                    "tile_size_range": [0.05, 0.5],
                    "tile_size_res": 7,
                    "stride_scaler": 0.5,
                    "image_shape": list(images.shape[2:4]),
                    "model_name": self.image_encoder.name,
                    "supersampling": self.config.supersampling,
                    "supersampling_factor": self.config.supersampling_factor,
                },
                cache_path=clip_cache_path,
                model=self.image_encoder,
            )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""

        self.train_count += 1
        
        # image_batch includes all training images
        image_batch = next(self.iter_train_image_dataloader) 
        
        cur_idx = step % image_batch['image_idx'].size(0)
        use_contrastive_loss = self.config.use_contrastive and step > self.config.contrastive_starting_epoch
        if use_contrastive_loss:
            new_image_batch = {}
            for k in image_batch.keys(): 
                new_image_batch[k] = image_batch[k][cur_idx:cur_idx+1]
            image_batch = new_image_batch
            
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)

        ray_indices = batch["indices"] # [image_idx, h, w]
        # print(ray_indices[:, 1].max())
        # print(ray_indices[:, 2].max())
        # print()
        
        batch['use_contrastive_loss'] = use_contrastive_loss
        if use_contrastive_loss:
            sample_idx = torch.randint(0, ray_indices.size(0), (1,))
            self.feature_interpolator.generate_mask(image_batch,batch["indices"][sample_idx])
            batch['sample_idx'] = sample_idx
            
        ray_bundle = self.train_ray_generator(ray_indices)
        

        if self.feature_type == 'clip':
            batch["feature"], clip_scale = self.feature_interpolator(ray_indices)
            ray_bundle.metadata["clip_scales"] = clip_scale
        elif self.feature_type == 'sam':
            batch["feature"], batch["mask"] = self.feature_interpolator(ray_indices, batch['use_contrastive_loss'])
        batch["dino"] = self.dino_dataloader(ray_indices)
       
        # assume all cameras have the same focal length and image width
        ray_bundle.metadata["fx"] = self.train_dataset.cameras[0].fx.item()
        ray_bundle.metadata["width"] = self.train_dataset.cameras[0].width.item()
        

        return ray_bundle, batch
    
    
    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch =  super().next_eval(step)
        return ray_bundle, batch
        