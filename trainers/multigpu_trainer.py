import wandb
import torch.nn as nn
from tqdm.auto import trange
import torch

from torch.utils.data import DataLoader
from pathlib import Path
from models.vgg import VGG


from torch.nn.parallel import DistributedDataParallel as DDP
import os
from utils.trainer_utils import (
    to_device,
    log_train_parameters,
    log_test_parameters,
    print_gpu_info,
)

SAVE_DIR = Path("./trained_models/")


class MultiGpuTrainer:
    def __init__(
        self,
        # TODO: add other possible model names here
        model: VGG,
        loaders: dict[str, DataLoader],
        criterion,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        # possible configs parameters
        gpu_id: int,
    ):
        self.model = model.to(gpu_id)

        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.gpu_id = gpu_id
        self.model = DDP(self.model, device_ids=[gpu_id])

    def train(self, max_epochs: int = 5, wandb_flag: bool = False):
        for epoch in trange(max_epochs):
            lr = self.optimizer.param_groups[0]["lr"]
            curr_test_acc = self.test(wandb_flag=wandb_flag)
            loss = self._run_epoch(epoch)

            print(f"Current test accuracy: {curr_test_acc:.2f}")
            print(f"Current loss: {loss:.2f}")

            if wandb_flag:
                log_train_parameters(
                    loss=loss, lr=lr, fracs_d=self.model.get_frac_orders(), epoch=epoch
                )

    def _run_epoch(self, epoch):
        loss = 0
        b_sz = len(next(iter(self.loaders["train"]))[0])
        print_gpu_info(epoch, self.gpu_id, b_sz, self.loaders["train"])

        for _, (data, target) in enumerate(self.loaders["train"]):
            data, target = to_device(data, target, device=self.gpu_id)
            self.optimizer.zero_grad()
            loss += self._run_batch(data, target)
        return loss

    def _run_batch(self, data, target):
        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, target)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def test(self, wandb_flag: bool = False):
        self.model.eval()

        with torch.no_grad():
            correct, total = 0, 0
            for data, target in self.loaders["test"]:
                data, target = to_device(data, target, device=self.gpu_id)
                total_b, correct_b = self.test_batch(data, target)
                total += total_b
                correct += correct_b

            test_acc = correct / total

            if wandb_flag:
                log_test_parameters(test_acc=test_acc)

        return test_acc

    def test_batch(self, data, target):
        outputs = self.model(data)
        _, predicted = torch.max(outputs.data, 1)
        total_b = target.size(0)
        correct_b = (predicted == target).sum().item()
        return correct_b, total_b
