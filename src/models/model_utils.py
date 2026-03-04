import torch
import torch.nn as nn
from robustbench import load_model


def load_from_robustbench(model: dict) -> nn.Module:
    model = load_model(model_name=model['name'], dataset='cifar10', threat_model='Linf')
    return model


def load_from_torch_hub(model: dict) -> nn.Module:
    model = torch.hub.load(model['source'], model['name'], pretrained=True)
    return model