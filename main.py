import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from robustbench.utils import load_model

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

import matplotlib.pyplot as plt

from src.models import ResNet18_CIFAR10


def visualize_adversarial_examples(x, x_adv, y_true, y_pred):
    # Visualize original and adversarial examples
    x = x * 255  # scale back to [0, 255] for visualization
    x_adv = x_adv * 255
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        axes[0, i].imshow(x[i].transpose(1, 2, 0).astype(np.uint8))
        axes[0, i].set_title(f"True: {y_true[i]}, Pred: {y_pred[i]}")
        axes[0, i].axis('off')

        axes[1, i].imshow(x_adv[i].transpose(1, 2, 0).astype(np.uint8))
        axes[1, i].set_title(f"Adversarial Example")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()


def train_surrogate_model() -> PyTorchClassifier:

    train_set = CIFAR10(
        './data',
        train=True,
        download=True,
        transform = ToTensor(),
    )
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    model = ResNet18_CIFAR10()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 1
    print(f'Training surrogate model for {epochs} epochs:', end='', flush=True)
    for epoch in range(epochs):
        print(".", end='', flush=True)
        # TODO, remove, just to see if everything works.
        i = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            i += 1
            if i >= 10:
                print("break early, to see if rest runs, remove this if training properly")
                break
    print("done.", flush=True)


    print("Surrogate model trained, saving to disk...", end='')
    torch.save(model.state_dict(), f'./models/surrogate/surrogate_model_e{epochs}.pt')
    print("saved.")

    return model


def main():

    os.makedirs('./models/surrogate', exist_ok=True)

    #------------------- Load data and surrogate model -----------------
    test_data = CIFAR10(
        './data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # load surrogate model from disk if it exists, otherwise train a new one and save it to disk for future use
    try:
        surrogate_model = ResNet18_CIFAR10()
        surrogate_model.load_state_dict(torch.load('./models/surrogate/surrogate_model_e1.pt', weights_only=True))

        print("Surrogate model loaded from disk.")
    except FileNotFoundError:
        print("Surrogate model not found on disk, training surrogate model.")
        surrogate_model = train_surrogate_model()


    #------------------- Evaluate surrogate model and create adversarial examples -----------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=0.01)

    # use ART Classifier wrapper to evaluate the surrogate model as ART expects it for attecks.
    surrogate_classifier = PyTorchClassifier(
        model=surrogate_model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        channels_first=True,
    )

    # use numpy arrays as ART works with them
    # TODO increase number of samples to evaluate on
    x, y = next(iter(test_loader)) # just check one batch
    y = y.detach().numpy()  # convert to numpy array as only used for accuracy calculation
    predictions = surrogate_classifier.predict(x.detach().numpy())
    accuracy = np.sum(np.argmax(predictions, axis=1) == y) / len(y)
    print("Accuracy surrogate model: {}%".format(accuracy * 100))

    attack = FastGradientMethod(estimator=surrogate_classifier, eps=0.2)
    x_test_adv = attack.generate(x.detach().numpy())

    predictions = surrogate_classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == y) / len(y)
    print("Accuracy on adversarial on surrogate model: {}%".format(accuracy * 100))


    #------------------- Evaluate robust model on clean and adversarial examples -----------------
    x_test_adv = torch.tensor(x_test_adv, dtype=torch.float32)  # needed as rest runs in torch
    # model is a torch model, it uses tensors.
    robust_model = load_model(model_name='Carmon2019Unlabeled', dataset="cifar10", threat_model="Linf")
    robust_model.eval()
    predictions_robust = robust_model(x)
    predictions_robust_adversarial = robust_model(x_test_adv)
    accuracy_robust = np.sum(np.argmax(predictions_robust.detach().numpy(), axis=1) == y) / len(y)
    accuracy_robust_adversarial = np.sum(np.argmax(predictions_robust_adversarial.detach().numpy(), axis=1) == y) / len(y)
    print("Accuracy on clean data on robust model: {}%".format(accuracy_robust * 100))
    print("Accuracy on adversarial data on robust model: {}%".format(accuracy_robust_adversarial * 100))
    visualize_adversarial_examples(x.detach().numpy(), x_test_adv.detach().numpy(), y, np.argmax(predictions_robust_adversarial.detach().numpy(), axis=1))


if __name__ == "__main__":
    main()