import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from robustbench import load_model

class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, normalize: bool = False):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def _normalize_fn(self, x : torch.Tensor) -> torch.Tensor:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]
        normalize = TF.normalize(x, mean, std)
        return normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = self._normalize_fn(x)
        return self.model(x)
    
class EnsembleWrapper(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = [m(x) for m in self.models]
        return torch.stack(logits).mean(dim=0)

def load_from_robustbench(
    model_name: str,
    model_dir: str = "./models",
    device: torch.device | str = "cuda",
) -> nn.Module:
    model = load_model(
        model_name=model_name,
        model_dir=model_dir,
        dataset="cifar10",
        threat_model="Linf",
    )
    model = model.to(device).eval()
    return model

def load_from_torch_hub(
    model_name: str,
    device: torch.device | str = "cuda",
) -> nn.Module:
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        model_name,
        pretrained=True,
    )
    model = model.to(device).eval() # pyright: ignore[reportAttributeAccessIssue]
    return model

def load_surrogate(
    cfg: dict,
    model_dir: str = "./models",
    device: torch.device | str = "cuda",
) -> tuple[nn.Module, bool]:
    source = cfg["source"]
    name = cfg["name"]
    normalize = False

    if source == "robustbench":
        model = load_from_robustbench(name, model_dir=model_dir, device=device)
    elif source == "chenyaofo":
        model = load_from_torch_hub(name, device=device)
    else:
        raise ValueError(f"Unknown source: {source}")

    if cfg.get("needs_normalize", False):
        normalize = True

    return model, normalize

def load_target(
    cfg: dict,
    model_dir: str = "./models",
    device: torch.device | str = "cuda",
) -> nn.Module:
    source = cfg["source"]
    name = cfg["name"]

    if source == "robustbench":
        model = load_from_robustbench(name, model_dir=model_dir, device=device)
    else:
        raise ValueError(f"Unknown target source: {source}")

    return model

def build_ensemble(
    surrogate_configs: list[dict],
    model_dir: str = "./models",
    device: torch.device | str = "cuda",
) -> tuple[nn.Module, None]:
    wrapped_models = []
    for cfg in surrogate_configs:
        model, normalize = load_surrogate(cfg, model_dir=model_dir, device=device)
        wrapped = ModelWrapper(model, normalize)
        wrapped_models.append(wrapped)

    ensemble = EnsembleWrapper(wrapped_models).to(device).eval()
    return ensemble, None