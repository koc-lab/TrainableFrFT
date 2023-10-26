from tqdm.auto import trange
import torch

from torch.utils.data import DataLoader
from pathlib import Path
from models.vgg import VGG
import copy
import wandb


from utils.trainer_utils import (
    to_device,
)


class Trainer:
    def __init__(
        self,
        # TODO: add other possible model names here as typing
        model,
        loaders: dict[str, DataLoader],
        criterion,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        # possible configs parameters
        gpu_id: int,
    ):
        # self.model = model.to(gpu_id)

        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)

        self.checkpoint = {}

    def pipeline(
        self,
        max_epochs: int,
        patience: int,
        wandb_flag: bool = False,
        sweep_id: str = None,
        early_stop_verbose: bool = False,
    ):
        early_stopping = EarlyStopping(patience=patience, verbose=early_stop_verbose)

        for epoch in trange(max_epochs):
            self.model.train()
            e_loss = self._run_epoch(data_key="train")
            test_acc = self.test(data_key="test")

            if wandb_flag:
                self.epoch_wandb_log(
                    loss=e_loss,
                    lr=self.optimizer.param_groups[0]["lr"],
                    
                    fracs_d=self.model.get_frac_orders(),
                    epoch=epoch,
                    test_acc=test_acc,
                )

            best_test_acc, best_model = early_stopping(test_acc, self.model, epoch)
            if early_stopping.early_stop:
                break

        self.pipeline_wandb_log(best_test_acc, best_model, sweep_id)
        self.sanity_check(best_model, best_test_acc)

    def _run_epoch(self, data_key: str = "train"):
        e_loss = 0
        for _, (data, target) in enumerate(self.loaders[data_key]):
            data, target = to_device(data, target, device=self.gpu_id)
            e_loss += self._run_batch(data, target)
        return e_loss

    def _run_batch(self, data, target):
        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, target)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        b_loss = loss.item()
        return b_loss

    def test(self, data_key: str = "test"):
        self.model.eval()

        with torch.no_grad():
            correct, total = 0.0, 0.0
            for data, target in self.loaders[data_key]:
                data, target = to_device(data, target, device=self.gpu_id)
                correct_b, total_b = self.test_batch(data, target)
                total += total_b
                correct += correct_b

            acc = correct / total
            return 100.0 * acc

    def test_batch(self, data, target):
        outputs = self.model(data)
        _, predicted = torch.max(outputs.data, 1)
        total_b = target.size(0)
        correct_b = (predicted == target).sum().item()
        return correct_b, total_b

    def sanity_check(self, best_model, best_test_acc):
        """
        A sanity check for the test accuracy of the best model, use it if you want to be sure
        """
        self.model = copy.deepcopy(best_model)
        acc_new = self.test(data_key="test")
        print(f"Best test accs: {best_test_acc}, {acc_new}")

    def save_checkpoint(self, test_acc: float, sweep_id: str):
        test_acc = str(round(test_acc, 5)).replace(".", "_")

        MODELS_DIR = Path(__file__).parent.parent.joinpath("model_checkpoints")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        SWEEP_ID_FOLDER = MODELS_DIR.joinpath(sweep_id)
        SWEEP_ID_FOLDER.mkdir(parents=True, exist_ok=True)
        FILE_NAME = f"{self.model.model_name}-{test_acc}-{wandb.run.id}-ckpt.pth"

        ckpt_path = Path.joinpath(MODELS_DIR, SWEEP_ID_FOLDER, FILE_NAME)
        torch.save(self.checkpoint, ckpt_path)

    def epoch_wandb_log(self, loss, lr, fracs_d, epoch, test_acc):
        wandb.log(data={"train/loss": loss}, step=epoch)
        wandb.log(data={"train/lr": lr}, step=epoch)
        wandb.log(data={"test/test_accuracy": test_acc}, step=epoch)
        try:
            for k, v in fracs_d.items():
                order1, order2 = v
                wandb.log(data={f"train/{k}_order1": order1}, step=epoch)
                wandb.log(data={f"train/{k}_order2": order2}, step=epoch)

        except AttributeError:
            print("Model does not have frac_order parameter")

    def pipeline_wandb_log(self, best_test_acc, best_model, sweep_id: str):
        self.checkpoint["test_acc"] = best_test_acc
        self.checkpoint["model"] = best_model
        self.checkpoint["model_state_dict"] = best_model.state_dict()
        self.save_checkpoint(best_test_acc, sweep_id)
        wandb.log(data={"test/best_test_acc": best_test_acc})


class EarlyStopping:
    def __init__(self, patience: int, verbose: bool = False):
        self.patience = patience
        self.verbose = verbose

        self.counter = 0
        self.best_test_acc = 0
        self.best_model = None
        self.early_stop = False

    def __call__(self, test_acc: float, model, epoch: int):
        if test_acc > self.best_test_acc:
            self.counter = 0
            self.best_test_acc = test_acc
            self.best_model = copy.deepcopy(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping at epoch {epoch}")

        return self.best_test_acc, self.best_model
