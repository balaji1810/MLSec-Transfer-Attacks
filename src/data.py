import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


def load_cifar10_testset(
    data_dir: str = "./data",
    num_samples: int | None = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=transforms.ToTensor(),
    )

    all_images = torch.stack([img for img, _ in dataset])  # (10000, 3, 32, 32)
    all_labels = torch.tensor([lbl for _, lbl in dataset])  # (10000,)

    if num_samples is not None and num_samples < len(all_labels):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(all_labels), size=num_samples, replace=False)
        indices.sort()
        indices = torch.from_numpy(indices)
        all_images = all_images[indices]
        all_labels = all_labels[indices]

    return all_images, all_labels
