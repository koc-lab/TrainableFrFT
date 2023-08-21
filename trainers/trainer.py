from tqdm.auto import trange
import torch

from torch.utils.data import DataLoader
from pathlib import Path
from models.vgg import VGG
import copy
import wandb


from utils.trainer_utils import (
    to_device,
    log_train_parameters,
    log_test_parameters,
)


# TODO: early stopping feature while training
# answer: this is done with sweep hopefully
# TODO: save the best model based on test accuracy
# answer: done in trainer pipeline function
# TODO: log the necessary artifacts, only when test acc increased
# TODO: check whether test acc in wandb table corresponds to last value or max of test/test_accuracy logs
# answer: it corresponds to last acc
# TODO: wandb count parametresi nasıl çalışıyor anla
# TODO: emirhanla neleri sweep edeceğimize karar verelim
# TODO: multiple gpu ile çalışırken size problemi çıkıyor, bunu çöz
# TODO: should I add optimizer state dict also to checkpoint?

# TODO: use inheritance for multigpu trainer
# TODO: YAML file dan config loadlayınca parametreleri resolve edemiyor pylance, tunaya sor
# TODO: sweep file içinde runları kaydet


class Trainer:
    def __init__(
        self,
        # TODO: add other possible model names here as typing
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

        self.checkpoint = {}

    def pipeline(
        self,
        max_epochs: int,
        patience: int,
        wandb_flag: bool = False,
        sweep_id: str = None,
    ):
        best_test_acc = 0.0
        best_model = None
        patience_ct = 0

        for epoch in trange(max_epochs):
            self.model.train()
            e_loss = self._run_epoch(data_key="train")
            test_acc = self.test(data_key="test")

            if test_acc > best_test_acc:
                patience_ct = 0
                best_test_acc = test_acc
                best_model = copy.deepcopy(self.model)
            else:
                patience_ct += 1

            if patience_ct == patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if wandb_flag:
                log_train_parameters(
                    loss=e_loss,
                    lr=self.optimizer.param_groups[0]["lr"],
                    fracs_d=self.model.get_frac_orders(),
                    epoch=epoch,
                )
                log_test_parameters(test_acc=test_acc, epoch=epoch)

        self.foo(best_test_acc, best_model, sweep_id)

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

    def _sanity_check_for_test_acc(self, best_model, best_test_acc):
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

        ckpt_path = Path.joinpath(
            MODELS_DIR,
            SWEEP_ID_FOLDER,
            f"{self.model.model_name}-{test_acc}-{wandb.run.id}-ckpt.pth",
        )

        torch.save(self.checkpoint, ckpt_path)

    def foo(self, best_test_acc, best_model: VGG, sweep_id: str):
        self.checkpoint["test_acc"] = best_test_acc
        self.checkpoint["model"] = best_model
        self.checkpoint["model_state_dict"] = best_model.state_dict()
        self.save_checkpoint(best_test_acc, sweep_id)
        wandb.log(data={"test/best_test_acc": best_test_acc})
