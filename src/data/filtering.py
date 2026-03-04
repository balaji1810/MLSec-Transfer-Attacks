import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def get_correct_indices(
    model,
    dataset,
    batch_size: int = 4,
    device: torch.device = torch.device("cuda"),
    num_workers: int = 2,
) -> list[int]:

    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    correct_indices = []
    idx_offset = 0

    for images, labels in tqdm(loader, desc="Filtering correct samples"):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        correct_mask = preds.eq(labels)
        batch_indices = torch.nonzero(correct_mask).squeeze(1)

        for i in batch_indices:
            correct_indices.append(idx_offset + i.item())

        idx_offset += images.size(0)

    return correct_indices


@torch.no_grad()
def filter_by_multiple_models(
    models: dict[str, torch.nn.Module],
    dataset,
    batch_size: int = 4,
    device: torch.device = torch.device("cuda"),
    num_workers: int = 2,
    mode: str = "intersection",
) -> list[int]:

    model_correct_sets = {}

    for name, model in models.items():
        print(f"Filtering for model: {name}")
        correct_indices = get_correct_indices(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
        )
        model_correct_sets[name] = set(correct_indices)

    if mode == "intersection":
        filtered = set.intersection(*model_correct_sets.values())
    elif mode == "union":
        filtered = set.union(*model_correct_sets.values())
    else:
        raise ValueError("mode must be 'intersection' or 'union'")

    print(f"Total filtered samples ({mode}): {len(filtered)}")
    return sorted(list(filtered))


def save_indices(indices: list[int], save_path: str):
    with open(save_path, "w") as f:
        json.dump({"indices": indices}, f, indent=4)

    print(f"Saved {len(indices)} indices to {save_path}")


def load_indices(path: str) -> list[int]:
    with open(path, "r") as f:
        data = json.load(f)

    return data["indices"]