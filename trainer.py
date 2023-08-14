import wandb
import torch.nn as nn
from tqdm.auto import trange
import torch

from torch.utils.data import DataLoader
from configurations.configs import TrainerConfig
from pathlib import Path
from models.vgg import VGG


SAVE_DIR = Path("./trained_models/")


class Trainer:
    def __init__(
        self,
        # TODO: add other possible model names here
        model: VGG,
        loaders: dict[str, DataLoader],
        criterion,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        config: TrainerConfig,
    ):
        self.model = model
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

    def train(self, wandb_flag: bool = False):
        # if wandb_flag:
        # wandb.watch(self.model, self.criterion, log="all", log_freq=10)

        batch_ct = 0
        for epoch in trange(self.config.epochs):
            for _, (images, labels) in enumerate(self.loaders["train"]):
                images, labels = self.to_device(images, labels)
                loss = self.train_batch(images, labels)
                # TODO: preserve batch_ct for future use, if n_epochs logging is not enough
                batch_ct += 1

            if wandb_flag:
                # TODO: get frac orders from model
                fracs_d = self.model.get_frac_orders()

                self.train_log(
                    loss=loss,
                    lr=self.optimizer.param_groups[0]["lr"],
                    fracs_d=fracs_d,
                    epoch=epoch,
                )

                self.log_model()

    def train_batch(self, images, labels):
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def test(self, wandb_flag: bool = False):
        # TODO: custom dataset yazıp bu total sample sayısına falan önceden hakim olmak daha iyi olur,
        # TODO: bir daha for loop içinde bir şeyler yapmayı çıkarmak lazım

        self.model.eval()

        with torch.no_grad():
            correct, total = 0, 0
            for images, labels in self.loaders["test"]:
                images, labels = self.to_device(images, labels)
                total_b, correct_b = self.test_batch(images, labels)
                total += total_b
                correct += correct_b

            if wandb_flag:
                wandb.log({"test/test_accuracy": correct / total})

    def test_batch(self, images, labels):
        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_b = labels.size(0)
        correct_b = (predicted == labels).sum().item()

        return correct_b, total_b

    def train_log(self, loss: float, lr: float, fracs_d: dict, epoch: int):
        wandb.log(data={"train/loss": loss}, step=epoch)
        wandb.log(data={"train/lr": lr}, step=epoch)

        try:
            for k, v in fracs_d.items():
                order1, order2 = v
                wandb.log(data={f"train/{k}_order1": order1}, step=epoch)
                wandb.log(data={f"train/{k}_order2": order2}, step=epoch)

        except AttributeError:
            print("Model does not have frac_order parameter")

    def test_log(self, correct, total):
        print(f"Accuracy of the model on the {total} test images: {correct / total:%}")
        wandb.log({"test/test_accuracy": correct / total})

    def log_model(self):
        # TODO: check why pylance is complaining model.config.model_name
        ckpt_file = SAVE_DIR / f"model_{self.model.config.model_name}.pth"
        torch.save(self.model.state_dict(), ckpt_file)

        # TODO: generate train sample images and log them to wandb
        artifact_name = f"{wandb.run.id}_{self.model.config.model_name}"
        at = wandb.Artifact(artifact_name, type="model")
        at.add_file(ckpt_file)
        wandb.log_artifact(at)

    def to_device(self, images, labels):
        return images.to(self.config.device), labels.to(self.config.device)
