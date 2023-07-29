import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from samnerf_v2.data.lerf_datamanager import (
    LERFDataManager,
    LERFDataManagerConfig,
)
from samnerf_v2.samnerf import LERFModel, LERFModelConfig

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

from samnerf_v2.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder
from samnerf_v2.encoders.clip_encoder import CLIPNetworkConfig
from samnerf_v2.encoders.openclip_encoder import OpenCLIPNetworkConfig
from samnerf_v2.encoders.sam_encoder import SAMNetworkConfig


@dataclass
class LERFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: LERFPipeline)
    """target class to instantiate"""
    datamanager: LERFDataManagerConfig = LERFDataManagerConfig()
    """specifies the datamanager config"""
    model: LERFModelConfig = LERFModelConfig()
    """specifies the model config"""
    feature_type: Literal['clip', 'xdecoder', 'sam'] = 'clip'
    """specifies the vision-language network config""" 
    use_contrastive: bool = False


class LERFPipeline(VanillaPipeline):
    def __init__(
        self,
        config: LERFPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode


        self.config.datamanager.feature_type = self.config.feature_type
        self.config.model.feature_type = self.config.feature_type
        
        self.config.datamanager.use_contrastive = self.config.use_contrastive
        self.config.model.use_contrastive = self.config.use_contrastive

        if self.config.feature_type == 'clip':
            network_config=OpenCLIPNetworkConfig(
                clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512
            ),
            #  You can swap the type of input encoder by specifying different NetworkConfigs, the one below uses OpenAI CLIP, the one above uses OpenCLIP
            # network=CLIPNetworkConfig(
            #     clip_model_type="ViT-B/16", clip_n_dims=512
            # )
        elif self.config.feature_type == 'sam':
            network_config=SAMNetworkConfig()
        else:
            raise ValueError("Invalid Feature Type.")

        # datamanager_config=LERFDataManagerConfig(
        #     dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
        #     train_num_rays_per_batch=4096,
        #     eval_num_rays_per_batch=4096,
        #     camera_optimizer=CameraOptimizerConfig(
        #         mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
        #     ),
        #     feature_type=self.config.feature_type
        # ),
        
        # model_config=LERFModelConfig(
        #     eval_num_rays_per_chunk=1 << 15,
        #     # NOTE: exceeding 16 layers per hashgrid causes a segfault within Tiny CUDA NN, so instead we compose multiple hashgrids together
        #     hashgrid_sizes=(19, 19),
        #     hashgrid_layers=(12, 12),
        #     hashgrid_resolutions=((16, 128), (128, 512)),
        #     num_lerf_samples=24,
        #     feature_type=self.config.feature_type
        # ),
        
        self.image_encoder: BaseImageEncoder = network_config.setup()
        

        self.datamanager: LERFDataManager = self.config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            image_encoder=self.image_encoder,
        )
        self.datamanager.to(device)

        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = self.config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            image_encoder=self.image_encoder,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(LERFModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])
