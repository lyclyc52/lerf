"""
LERF configuration file.
"""

from nerfstudio.configs.base_config import ViewerConfig

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from samnerf_v2.data.sam_datamanager import SAMDataManagerConfig
from samnerf_v2.samnerf import LERFModelConfig
from samnerf_v2.samnerf_pipeline import LERFPipelineConfig
from samnerf_v2.samnerf_trainer import SAMNERFTrainerConfig, SAMNERFTrainer
# from samnerf_v2.samnerf_viewer import SAMNERFViewerConfig
"""
Swap out the network config to use OpenCLIP or CLIP here.
"""


samnerf_method = MethodSpecification(
    config=SAMNERFTrainerConfig(
        method_name="samnerf-v2",
        steps_per_save=500,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=LERFPipelineConfig(
            datamanager=SAMDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                )
            ),
            model=LERFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # NOTE: exceeding 16 layers per hashgrid causes a segfault within Tiny CUDA NN, so instead we compose multiple hashgrids together
                hashgrid_sizes=(19, 19),
                hashgrid_layers=(12, 12),
                hashgrid_resolutions=((16, 128), (128, 512)),
                num_lerf_samples=24
            ),
            feature_type='clip',
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "lerf": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=4000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        # viewer=SAMNERFViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for LERF",
)

# lerf_method_big = MethodSpecification(
#     config=TrainerConfig(
#         method_name="lerf-big",
#         steps_per_eval_batch=500,
#         steps_per_save=2000,
#         max_num_iterations=30000,
#         mixed_precision=True,
#         pipeline=LERFPipelineConfig(
#             datamanager=LERFDataManagerConfig(
#                 dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
#                 train_num_rays_per_batch=8192,
#                 eval_num_rays_per_batch=4096,
#                 camera_optimizer=CameraOptimizerConfig(
#                     mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
#                 ),
#             ),
#             model=LERFModelConfig(
#                 eval_num_rays_per_chunk=1 << 15,
#                 # NOTE: exceeding 16 layers per hashgrid causes a segfault within Tiny CUDA NN, so instead we compose multiple hashgrids together
#                 hashgrid_sizes=(19, 19),
#                 hashgrid_layers=(16, 16),
#                 hashgrid_resolutions=((16, 128), (128, 512)),
#                 num_lerf_samples=32,
#             ),
#             network=OpenCLIPNetworkConfig(
#                 clip_model_type="ViT-L-14", clip_model_pretrained="laion2b_s32b_b82k", clip_n_dims=768
#             ),
#         ),
#         optimizers={
#             "proposal_networks": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": None,
#             },
#             "fields": {
#                 "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
#             },
#             "lerf": {
#                 "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=3000),
#             },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="viewer",
#     ),
#     description="A larger version of LERF with a higher memory footprint, bigger CLIP model, and more hashgrid capacity",
# )

# lerf_method_lite = MethodSpecification(
#     config=TrainerConfig(
#         method_name="lerf-lite",
#         steps_per_eval_batch=500,
#         steps_per_save=2000,
#         max_num_iterations=30000,
#         mixed_precision=True,
#         pipeline=LERFPipelineConfig(
#             datamanager=LERFDataManagerConfig(
#                 dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
#                 train_num_rays_per_batch=4096,
#                 eval_num_rays_per_batch=4096,
#                 camera_optimizer=CameraOptimizerConfig(
#                     mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
#                 ),
#             ),
#             model=LERFModelConfig(
#                 eval_num_rays_per_chunk=1 << 15,
#                 hashgrid_sizes=(19,),
#                 hashgrid_layers=(16,),
#                 hashgrid_resolutions=((16, 512),),
#                 num_lerf_samples=12,
#             ),
#             network=OpenCLIPNetworkConfig(
#                 clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512
#             ),
#         ),
#         optimizers={
#             "proposal_networks": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": None,
#             },
#             "fields": {
#                 "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
#             },
#             "lerf": {
#                 "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=7000),
#             },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="viewer",
#     ),
#     description="A lightweight version of LERF designed to work on smaller GPUs",
# )
