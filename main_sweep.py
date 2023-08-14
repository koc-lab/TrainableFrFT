# %%
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch.nn as nn
import wandb
from utils.utils import set_seeds, get_device

device = "cpu"
from dataloader import make_loaders
from trainer import Trainer
from torch.optim import SGD
from configurations.configs import TrainerConfig, VGGConfig, SweepConfig
from models.vgg import VGG

# # Sweep

set_seeds()

parameters_dict = SweepConfig(
    # grid search
    model_name=dict(values=["VGG_11", "VGG_11_FRFT", "VGG_11_DFRFT", "VGG_11_FFT"]),
    # random search
    learning_rate=dict(distribution="uniform", min=0.0005, max=0.5),
    batch_size=dict(distribution="q_log_uniform_values", q=8, min=32, max=256),
    # constant parameters
    epochs=dict(value=5),
    dataset=dict(value="CIFAR-10"),
    device=dict(value=device),
    # constant trainer parameters
    classes=dict(value=10),
    weight_decay=dict(value=5e-4),
    momentum=dict(value=0.9),
)

# %%
parameters_dict = parameters_dict.__dict__


def run_sweep(config: SweepConfig = None):
    # tell wandb to get started
    run = wandb.init(
        project="demo-sweep-run-v3",
        config=config,
        # TODO: fill the None values
        job_type=None,
        name=None,
    )
    config = wandb.config
    # above line enables to acces the keys of config as attributes

    trainer_config = TrainerConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
        device=config.device,
    )

    model_config = VGGConfig(model_name=config.model_name, n_class=config.classes)

    # make the model, data, and optimization problem
    model, loaders, criterion, optimizer, scheduler, trainer_config = make(
        trainer_config, model_config
    )

    model_trainer = Trainer(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_config=trainer_config,
    )

    # train
    model_trainer.train(wandb_flag=True)

    # test
    model_trainer.test(wandb_flag=True)

    return model


import torch


def make(trainer_config: TrainerConfig, model_config: VGGConfig):
    # Make the data

    loaders = make_loaders(
        batch_size=trainer_config.batch_size, train_slice=1000, test_slice=100
    )

    model = VGG(config=model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=trainer_config.learning_rate,
        momentum=trainer_config.momentum,
        weight_decay=trainer_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=trainer_config.epochs
    )

    return model, loaders, criterion, optimizer, scheduler, trainer_config


#  %%
sweep_config = dict(
    method="random",
    # TODO:  ensure that you log test/test_accuracy
    metric=dict(name="test/test_accuracy", goal="maximize"),
    parameters=parameters_dict,
)

import pprint

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="demo-sweep-run-v3")
wandb.agent(sweep_id, run_sweep, count=None)
