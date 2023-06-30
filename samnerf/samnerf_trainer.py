from typing import Literal
from nerfstudio.engine.trainer import Trainer, TrainerConfig
import torch
from dataclasses import dataclass, field
from typing import Literal, Type


@dataclass
class SAMNERFTrainerConfig(TrainerConfig):
    """Configuration for training regimen"""
    _target: Type = field(default_factory=lambda: SAMNERFTrainer)
    
    
class SAMNERFTrainer(Trainer):
    
    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        super().__init__(config, local_rank, world_size)
        
        
    def setup(self, test_mode: Literal['test', 'val', 'inference'] = "val") -> None:

        super().setup(test_mode)
        for _, param in self.pipeline._model.named_parameters():
            param.data = param.to(torch.float32)