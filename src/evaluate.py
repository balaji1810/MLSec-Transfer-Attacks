import torch
import torch.nn as nn


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    images: torch.Tensor,
    batch_size: int = 64,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    model.eval()
    preds_list = []
    n = len(images)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = images[start:end].to(device)
        logits = model(x_batch)
        preds_list.append(logits.argmax(dim=1).cpu())

    return torch.cat(preds_list, dim=0)


def evaluate_transfer(
    target_model: nn.Module,
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 64,
    device: torch.device | str = "cuda",
) -> dict:
    # Predictions
    clean_preds = get_predictions(target_model, clean_images, batch_size, device)
    adv_preds = get_predictions(target_model, adv_images, batch_size, device)

    n = len(labels)

    # Clean accuracy
    correct_clean = (clean_preds == labels)
    clean_acc = correct_clean.float().mean().item() * 100.0

    # Adversarial accuracy (over all samples)
    correct_adv = (adv_preds == labels)
    adv_acc = correct_adv.float().mean().item() * 100.0

    # Attack success rate (conditional on clean-correct)
    num_correct_clean = correct_clean.sum().item()
    if num_correct_clean > 0:
        fooled = correct_clean & (~correct_adv)
        asr = fooled.sum().item() / num_correct_clean * 100.0
    else:
        asr = 0.0

    # Perturbation statistics
    diff = (adv_images - clean_images).abs()
    # Per-sample Linf
    per_sample_linf = diff.view(n, -1).max(dim=1).values
    mean_linf = per_sample_linf.mean().item()
    max_linf = per_sample_linf.max().item()

    return {
        "num_samples": n,
        "clean_accuracy": round(clean_acc, 2),
        "adversarial_accuracy": round(adv_acc, 2),
        "attack_success_rate": round(asr, 2),
        "mean_linf_perturbation": round(mean_linf, 6),
        "max_linf_perturbation": round(max_linf, 6),
    }
