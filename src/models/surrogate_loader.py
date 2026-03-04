"""
find model id's at https://github.com/chenyaofo/pytorch-cifar-models
"""
from .model_wrapper import EnsembleWrapper
from .model_utils import load_from_robustbench, load_from_torch_hub


def load_surrogate_models(surrogate_models: list[dict], device: str, **kargs) -> EnsembleWrapper:
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

    return EnsembleWrapper(surrogates=loaded_surrogate_models, normalize=normalize_filter).to(device)