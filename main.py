import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from src.data.filtering import get_correct_indices, save_indices
from robustbench.utils import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

batch_size = 4
transform = transforms.Compose([transforms.ToTensor()])
testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

surrogate_1 = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf').to(device).eval()

indices = get_correct_indices(
    model=surrogate_1,
    dataset=testset,
    device=device,
    batch_size=batch_size,
)
print(f"Length of correct indices : {len(indices)}")
save_indices(indices, save_path="./data/cifar10_correct_indices.json")


