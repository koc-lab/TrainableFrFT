import wandb
from torch.optim import Adam, SGD

from configurations.configs import OptimizerConfig, SchedulerConfig
from configurations.configs import OptimizerType, SchedulerType

from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR
import torchvision.transforms as transforms
import torch


def get_optimizer(config: OptimizerConfig, model: torch.nn.Module):
    if config.optimizer_type is OptimizerType.SGD:
        return SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.wd,
        )
    elif config.optimizer_type is OptimizerType.Adam:
        return Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.wd,
        )


def get_scheduler(config: SchedulerConfig, optimizer, max_epochs: int):
    if config.scheduler_type is SchedulerType.CosineAnnealingLR:
        return CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif  config.scheduler_type is SchedulerType.StepLR:
        return  StepLR(optimizer, step_size=25, gamma=0.1)


def get_wandb_config_dict(**kwargs):
    return kwargs


def wandb_single_run_setup(project_name: str, **kwargs):
    config_dict = get_wandb_config_dict(**kwargs)
    run = wandb.init(project=project_name, config=config_dict, job_type=None, name=None)
    config = wandb.config
    return run, config
