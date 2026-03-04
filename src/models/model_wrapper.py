import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

def make_normalize_fn(x : torch.Tensor) -> torch.Tensor:
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    normalize = TF.normalize(x, mean, std)
    return normalize


def identity_preprocess(x: torch.Tensor) -> torch.Tensor:
    return x


class ModelWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        preprocess_fn: str | None = "normalize",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocess_fn = (make_normalize_fn if preprocess_fn == "normalize" else identity_preprocess)
        

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x_p = self.preprocess_fn(x)
        logits = self.model(x_p)
        return logits.detach().cpu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_p = self.preprocess_fn(x)
        logits = self.model(x_p)
        return logits
    

class EnsembleWrapper(nn.Module):
    def __init__(
        self,
        surrogates : list[nn.Module],
        standard_model : nn.Module,
    ):
        super().__init__()
        self.surrogates = nn.ModuleList(surrogates)
        self.standard_model = standard_model
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2471, 0.2435, 0.2616]

    def forward(self, x):
        x_norm = TF.normalize(x, mean=self.mean, std=self.std)
        
        surrogate_outputs = [m(x_norm) for m in self.surrogates]

        standard_logits = self.standard_model(x)
        outputs = surrogate_outputs + [standard_logits]

        
        return outputs