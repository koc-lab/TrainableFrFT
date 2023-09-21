import pickle
import torch
import numpy as np
import random
from configurations.configs import PoolType


def set_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def combine_dataclass_attributes(instance_a, instance_b):
    dict_a = instance_a.__dict__
    dict_b = instance_b.__dict__

    combined_dict = {**dict_a, **dict_b}
    return combined_dict


def change_pooling(input_list, to=PoolType.FrFTPool):
    result = []
    for i in range(len(input_list)):
        item = input_list[i]
        if i == len(input_list) - 1:
            result.append(item)
            break
        if item is PoolType.MaxPool:
            result.append(to)
        else:
            result.append(item)
    return result


def dump_model_variants_dict(
    model_name: str, model_variants_dict: dict[str, list]
) -> None:
    """
    Save a dictionary of model variants to a pickle file.

    This function takes a model name and a dictionary containing model variants as input,
    and it saves the dictionary as a pickle file named based on the model name.

    Args:
        model_name (str): The name of the model (VGG, ResNet, etc.).
        model_variants_dict (dict): A dictionary containing model variants as keys and their details as values.

    Returns:
        None
    """

    filename = f"model_dicts/{model_name}_variants.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model_variants_dict, f)
        


def get_model_variants_dict(model_name: str) -> dict[str, list]:
    file_name = f"model_dicts/{model_name}_variants.pkl"

    with open(file_name, "rb") as pickle_file:
        loaded_dict = pickle.load(pickle_file)

    return loaded_dict
