# %%
from dotenv import load_dotenv

load_dotenv()


import torch
from models.vgg import VGG
import wandb
import yaml
from utils.utils import set_seeds

set_seeds()

from trainers.trainer import Trainer
from utils.data_utils import CustomDataHandler, get_basic_transform

from configurations.configs import (
    OptimizerType,
    SchedulerType,
    DataHandlerConfig,
    OptimizerConfig,
    SchedulerConfig,
)

from utils.main_utils import get_optimizer, get_scheduler


with open("configurations/sweep_config.yml") as file:
    sweep_config = yaml.load(file, Loader=yaml.FullLoader)


# %%


def run_sweep(config: dict = None):
    # tell wandb to get started
    global sweep_id

    run = wandb.init(config=config)
    config = wandb.config

    train_t, test_t = get_basic_transform()

    dh_config = DataHandlerConfig(
        batch_size=config.batch_size,
        multi_gpu=config.multi_gpu,
        train_slice=config.train_slice,
        test_slice=config.test_slice,
        train_transform=train_t,
        test_transform=test_t,
    )

    optimizer_config = OptimizerConfig(
        optimizer_type=OptimizerType.SGD,
        lr=config.lr,
        wd=config.wd,
        momentum=config.momentum,
    )

    scheduler_config = SchedulerConfig(
        scheduler_type=SchedulerType.CosineAnnealingLR,
        max_epochs=config.max_epochs,
    )

    custom_dataclass = CustomDataHandler(config=dh_config)
    loaders = custom_dataclass.loaders

    model = VGG(model_name=config.model_name, n_class=config.n_class)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        config=optimizer_config,
        model=model,
    )
    scheduler = get_scheduler(
        config=scheduler_config,
        optimizer=optimizer,
        max_epochs=config.max_epochs,
    )
    trainer = Trainer(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        gpu_id=config.gpu_id,
    )

    trainer.pipeline(
        max_epochs=config.max_epochs,
        patience=config.patience,
        wandb_flag=True,
        sweep_id=sweep_id,
        early_stop_verbose=config.early_stop_verbose,
    )


sweep_id = wandb.sweep(sweep_config, project="frft-demo")
wandb.agent(sweep_id, function=run_sweep, count=4)
