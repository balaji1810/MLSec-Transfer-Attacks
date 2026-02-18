import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# ResNet-18 with pretrained weights modified for CIFAR-10
class ResNet18_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_CIFAR10, self).__init__()
        # Load the pretrained ResNet-18 model
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept 3-channel input (CIFAR-10 images)
        # self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove the max pooling layer
        # self.model.maxpool = nn.Identity()

        # Modify the fully connected layer to output the correct number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


