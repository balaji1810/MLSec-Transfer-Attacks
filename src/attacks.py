import torch
import torch.nn as nn
import torchattack
from typing import Callable
from tqdm import tqdm

attacks: dict[str, type] = {
    "IFGSM": torchattack.IFGSM,
    "MIFGSM": torchattack.MIFGSM,
    "DIFGSM": torchattack.DIFGSM,
    "TIFGSM": torchattack.TIFGSM,
    "SINIFGSM": torchattack.SINIFGSM,
    "Admix": torchattack.Admix,
    "VMIFGSM": torchattack.VMIFGSM,
}


def create_attack(
    attack_name: str,
    model: nn.Module,
    normalize: Callable | None,
    device: torch.device,
    eps: float,
    **kwargs,
) -> torchattack.Attack:
    atk_cls = attacks[attack_name]
    attack = atk_cls(
        model=model,
        normalize=normalize,
        device=device,
        eps=eps,
        clip_min=0.0,
        clip_max=1.0,
        **kwargs,
    )
    return attack


def generate_adversarial_examples(
    attack: torchattack.Attack,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 64,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    adv_images = torch.empty_like(images)
    n = len(images)

    if(attack.__class__.__name__ == "Admix"):
        print(f"Using {attack.__class__.__name__} attack, reducing batch size to save memory.")
        batch_size = int(batch_size / 3)

    for start in tqdm(range(0, n, batch_size), desc="Generating adversarial examples"):
        end = min(start + batch_size, n)
        x_batch = images[start:end].to(device)
        y_batch = labels[start:end].to(device)

        with torch.enable_grad():
            x_adv = attack(x_batch, y_batch)

        adv_images[start:end] = x_adv.detach()

    print(f"Shape of adversarial images: {adv_images.shape}")

    # clamp to [0,1]
    # adv_images = adv_images.clamp(0.0, 1.0)

    return adv_images
