# %%
import torch.nn as nn
from models.vgg import VGG
from dotenv import load_dotenv

load_dotenv()


from trainers.trainer import Trainer
from utils.utils import set_seeds
from configurations.configs import DataHandlerConfig, OptimizerConfig, SchedulerConfig
from configurations.configs import OptimizerType, SchedulerType
from utils.main_utils import get_optimizer, get_scheduler
from utils.main_utils import wandb_single_run_setup

from utils.data_utils import CustomDataHandler, get_basic_transform

set_seeds()

# TODO: Use dataclass approach for these parameters


def main(
    model_name: str,
    gpu_id: int,
    n_class: int = 10,
    max_epochs: int = 5,
    wandb_flag: bool = False,
    multi_gpu: bool = False,
):
    train_t, test_t = get_basic_transform()

    dh_config = DataHandlerConfig(
        batch_size=128,
        multi_gpu=multi_gpu,
        train_slice=1000,
        test_slice=100,
        train_transform=train_t,
        test_transform=test_t,
    )

    optimizer_config = OptimizerConfig(
        optimizer_type=OptimizerType.SGD, lr=0.1, wd=5e-4, momentum=0.9
    )

    scheduler_config = SchedulerConfig(
        scheduler_type=SchedulerType.CosineAnnealingLR, max_epochs=max_epochs
    )

    if wandb_flag:
        run, config = wandb_single_run_setup(
            project_name="frft-demo",
            batch_size=dh_config.batch_size,
            learning_rate=optimizer_config.lr,
            weight_decay=optimizer_config.wd,
            momentum=optimizer_config.momentum,
            model_name=model_name,
            n_class=n_class,
            train_slice=dh_config.train_slice,
            test_slice=dh_config.test_slice,
        )

    custom_dataclass = CustomDataHandler(config=dh_config)
    loaders = custom_dataclass.loaders

    model = VGG(model_name=model_name, n_class=n_class)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        config=optimizer_config,
        model=model,
    )
    scheduler = get_scheduler(
        config=scheduler_config,
        optimizer=optimizer,
        max_epochs=max_epochs,
    )
    trainer = Trainer(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        gpu_id=gpu_id,
    )

    trainer.train(max_epochs=max_epochs, wandb_flag=wandb_flag)


if __name__ == "__main__":
    model_name = "VGG_16"
    main(
        model_name=model_name,
        gpu_id=0,
        n_class=10,
        max_epochs=5,
        wandb_flag=True,
        multi_gpu=False,
    )

# %%
