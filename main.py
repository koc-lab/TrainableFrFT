# %%
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import wandb
from utils.utils import set_seeds, get_device, combine_dataclass_attributes

device = "cpu"
from dataloader import make_loaders
from trainer import Trainer
from torch.optim import SGD
from configurations.configs import TrainerConfig, VGGConfig
from models.vgg import VGG

set_seeds()
# Single run example

trainer_config = TrainerConfig(
    epochs=5,
    batch_size=64,
    learning_rate=0.5,
    weight_decay=5e-4,
    momentum=0.9,
    device=device,
)

model_config = VGGConfig(
    model_name="VGG_11_FRFT",
    n_class=10,
)

combined_config_dict = combine_dataclass_attributes(
    trainer_config,
    model_config,
)

# %%
wandb_flag = True

if wandb_flag:
    run = wandb.init(
        project="demo-single-run-v2",
        config=combined_config_dict,
        # TODO: fill the None values
        job_type=None,
        name=None,
    )
    config = wandb.config

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

trainer = Trainer(
    model=model,
    loaders=loaders,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    config=trainer_config,
)


# %%
# from models.custom_frft_layers import FrFTPool, DFrFTPool

# d = {}
# pool_count = 1
# for layer in model.vgg_block.layers:
#     if isinstance(layer, FrFTPool) or isinstance(layer, DFrFTPool):
#         order_1, order_2 = layer.order1.item(), layer.order2.item()
#         print(order_1, order_2)
#         d[f"pool_{pool_count}"] = (order_1, order_2)
#         pool_count += 1

# %%
trainer.train(wandb_flag=True)
trainer.test(wandb_flag=True)
