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


from nerfstudio.utils.decorators import (
    check_eval_enabled,
    check_main_thread,
    check_viewer_enabled,
)

class SAMOptimizers(Optimizers):
    def __init__(self, config: Dict[str, Any], param_groups: Dict[str, List[Parameter]]) -> None:
        super().__init__(config, param_groups)
        
        
    def load_schedulers(self, loaded_state: Dict[str, Any]) -> None:
        """Helper to load the scheduler state from previous checkpoint
        Args:
            loaded_state: the state from the previous checkpoint
        """
        for k, v in loaded_state.items():
            
            self.schedulers[k].load_state_dict(v)

@dataclass
class SAMNERFTrainerConfig(TrainerConfig):
    """Configuration for training regimen"""
    _target: Type = field(default_factory=lambda: SAMNERFTrainer)
    
    
class SAMNERFTrainer(Trainer):
    
    optimizers: SAMOptimizers
    
    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        super().__init__(config, local_rank, world_size)
        
        wandb.init(
            # set the wandb project where this run will be logged
            project="debug",
            
            # track hyperparameters and run metadata
            config={
            "type": "resume training"
            }
        )
        
    def setup_optimizers(self) -> Optimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        optimizer_config = self.config.optimizers.copy()
        param_groups = self.pipeline.get_param_groups()
        camera_optimizer_config = self.config.pipeline.datamanager.camera_optimizer
        if camera_optimizer_config is not None and camera_optimizer_config.mode != "off":
            assert camera_optimizer_config.param_group not in optimizer_config
            optimizer_config[camera_optimizer_config.param_group] = {
                "optimizer": camera_optimizer_config.optimizer,
                "scheduler": camera_optimizer_config.scheduler,
            }
        return SAMOptimizers(optimizer_config, param_groups)
    
    def setup(self, test_mode: Literal['test', 'val', 'inference'] = "val") -> None:
        super().setup(test_mode)
        for _, param in self.pipeline._model.named_parameters():
            param.data = param.to(torch.float32)
            
            
    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:

            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])

            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")


    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()
    
    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        self.optimizers.zero_grad_all()
        cpu_or_cuda_str: str = self.device.split(":")[0]
        
        _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
        loss = functools.reduce(torch.add, loss_dict.values())

        loss.backward()  # type: ignore
        self.optimizers.optimizer_step_all()

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)


        self.optimizers.scheduler_step_all(step)

        wandb.log(loss_dict)
        return loss, loss_dict, metrics_dict
