# %%
import torch.nn as nn
from models.vgg import VGG
from trainers.multigpu_trainer import MultiGpuTrainer
import torch
import torch.multiprocessing as mp
from utils.data_utils import CustomDataHandler

from utils.utils import set_seeds


from configurations.configs import DataHandlerConfig, OptimizerConfig, SchedulerConfig
from configurations.configs import OptimizerType, SchedulerType
from utils.main_utils import (
    get_optimizer,
    get_scheduler,
    get_basic_transform,
    wandb_single_run_setup,
)

from utils.trainer_utils import ddp_setup
from torch.distributed import destroy_process_group

set_seeds()


# TODO: Use dataclass approach for these parameters
def main(
    rank: int,
    world_size: int,
    max_epochs: int,
    multi_gpu: bool,
    wandb_flag: bool,
):
    ddp_setup(rank, world_size)
    model_name = "VGG_16_FRFT"
    n_class = 10

    dh_config = DataHandlerConfig(
        batch_size=128,
        multi_gpu=multi_gpu,
        train_slice=1,
        test_slice=1,
        transform=get_basic_transform(),
    )

    optimizer_config = OptimizerConfig(
        optimizer_type=OptimizerType.SGD, lr=0.1, wd=5e-4, momentum=0.9
    )

    scheduler_config = SchedulerConfig(
        scheduler_type=SchedulerType.CosineAnnealingLR, max_epochs=max_epochs
    )

    custom_dataclass = CustomDataHandler(config=dh_config)
    loaders = custom_dataclass.loaders

    model = VGG(model_name=model_name, n_class=n_class)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config=optimizer_config, model=model)
    scheduler = get_scheduler(
        config=scheduler_config, optimizer=optimizer, max_epochs=max_epochs
    )

    trainer = MultiGpuTrainer(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        gpu_id=rank,
    )

    trainer.train(max_epochs=max_epochs, wandb_flag=wandb_flag)
    destroy_process_group()


if __name__ == "__main__":
    max_epochs = 200
    multi_gpu = True
    wandb_flag = False

    world_size = torch.cuda.device_count()
    print(f"world_size: {world_size}")
    mp.spawn(
        main, args=(world_size, max_epochs, multi_gpu, wandb_flag), nprocs=world_size
    )


# %%
