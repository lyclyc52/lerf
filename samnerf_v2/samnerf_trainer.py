from typing import Literal
from nerfstudio.engine.trainer import *
from nerfstudio.engine.trainer import TRAIN_INTERATION_OUTPUT
import torch
from dataclasses import dataclass, field
from typing import Literal, Type, Any, Dict, List, Optional
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter
from pathlib import Path
from torch.nn.parameter import Parameter
import wandb

@dataclass
class SAMNERFTrainerConfig(TrainerConfig):
    """Configuration for training regimen"""
    _target: Type = field(default_factory=lambda: SAMNERFTrainer)
    use_wandb:bool = True
    
    load_origin_nerf_checkpoint:Optional[Path] = None
    
    base_dir:str = None
    
    def get_base_dir(self) -> Path:
        if self.base_dir is None:
            return super().get_base_dir()
        else:
            assert self.method_name is not None, "Please set method name in config or via the cli"
            self.set_experiment_name()
            return Path(f"{self.output_dir}/{self.experiment_name}/{self.method_name}/{self.base_dir}")
            
    
    
class SAMNERFTrainer(Trainer):
    def __init__(self, config: SAMNERFTrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        super().__init__(config, local_rank, world_size)
        
        if self.config.use_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="debug",
                # track hyperparameters and run metadata
                config={
                "type": "resume training"
                }
            )
    
    def setup(self, test_mode: Literal['test', 'val', 'inference'] = "val") -> None:
        return_value = super().setup(test_mode)
        self._load_origin_nerf_checkpoint()
        return return_value
    
    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        
        feature_starting_epoch = self.pipeline.datamanager.config.feature_starting_epoch
        
        if step == feature_starting_epoch:
            save_root = os.path.join(self.config.get_base_dir(), 'render_results')
            self.pipeline.load_feature(save_root)
        loss, loss_dict, metrics_dict = super().train_iteration(step)
        if self.config.use_wandb:
            wandb.log(loss_dict)
        return loss, loss_dict, metrics_dict
    
    
    def _load_origin_nerf_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_checkpoint = self.config.load_origin_nerf_checkpoint
 
        if load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_origin_nerf_pipeline(loaded_state["pipeline"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")


