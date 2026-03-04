"""
find model id's at https://github.com/RobustBench/robustbench?tab=readme-ov-file#cifar-10
"""

import torch.nn as nn

from .model_utils import load_from_robustbench


def load_all_targets(target_models: list[dict], device: str, **kwargs) -> dict[str, nn.Module]:
    """
    Load all target models specified in the config and return a dictionary of model name to model instance.
    """
    loaded_targets = {}
    for model in target_models:
        loaded_targets[model['name']] = load_from_robustbench(model).to(device)
    return loaded_targets