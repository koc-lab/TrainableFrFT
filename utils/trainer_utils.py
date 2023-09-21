# %%
import os
import re
import wandb
import torch
from models.vgg import VGG
from pathlib import Path
from torch.distributed import init_process_group


def str_to_acc(filename: str) -> float:
    pattern = r"(\d+_\d+)"
    match = re.search(pattern, filename)
    if match:
        extracted_value = match.group(1)
        float_value = float(extracted_value.replace("_", "."))
        return float_value


def to_device(data, target, device):
    return data.to(device), target.to(device)


def log_model_artifact(test_acc: str, ckpt_file: Path):
    artifact_name = f"{wandb.run.id}_{ckpt_file.stem}_{test_acc}"
    artifact = wandb.Artifact(name=artifact_name, type="Model")
    artifact.add_file(ckpt_file)
    wandb.log_artifact(artifact)


def get_existing_model(model_name: str, SAVE_DIR: Path):
    files_in_dir = os.listdir(SAVE_DIR)

    for file in files_in_dir:
        if file.startswith(model_name):
            best_acc = str_to_acc(filename=file)
            best_model = torch.load(SAVE_DIR / file)
            return best_acc, best_model


def remove_old_models(model_name: str, SAVE_DIR: Path):
    pass

    # TODO: finish this function


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def print_gpu_info(epoch, gpu_id, batch_size, train_data):
    print(
        f"GPU{gpu_id} | Epoch {epoch} | Batchsize: {batch_size} | Steps: {len(train_data)}"
    )
