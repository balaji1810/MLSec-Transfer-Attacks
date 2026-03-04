"""
find model id's at https://github.com/chenyaofo/pytorch-cifar-models
"""
import torch
import torch.nn as nn
from robustbench import load_model
from .model_wrapper import EnsembleWrapper


def load_from_robustbench(model: dict) -> nn.Module:
    model = load_model(model_name=model['name'], dataset='cifar10', threat_model='Linf')
    return model


def load_from_torch_hub(model: dict) -> nn.Module:
    model = torch.hub.load(model['source'], model['name'], pretrained=True)
    return model


def load_surrogate_models(surrogate_models: list[dict], **kargs) -> EnsembleWrapper:
    """
    If the config was loaded correctly, this should work fine:
    >>> surrogate_models = load_surrogate_models(**config)

    example config part:
    ```yml
    surrogate_models:
        - name: "standard"
          source: "robustbench"
          normalize: false
        - name: "cifar10_resnet20"
          source: "chenyaofo/pytorch-cifar-models"
          normalize: false
    ```
    """
    loaded_surrogate_models = []
    normalize_filter = []
    for model in surrogate_models:
        match model['source']:
            case 'robustbench':
                loaded_surrogate_models.append(load_from_robustbench(model))
                normalize_filter.append(model['normalize'])
            case 'chenyaofo/pytorch-cifar-models':
                loaded_surrogate_models.append(load_from_torch_hub(model))
                normalize_filter.append(model['normalize'])
            case _:
                raise ValueError(f"Unknown surrogate model source: {model['source']}")

    return EnsembleWrapper(surrogates=loaded_surrogate_models, normalize=normalize_filter)