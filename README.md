# Transfer-Based Adversarial Attack Study on CIFAR-10

A reproducible study of **transfer-based adversarial attacks** using non-robust surrogate models to attack robust CIFAR-10 models from [RobustBench](https://robustbench.github.io/).

## Overview

This project generates adversarial examples on **surrogate models** (non-robust) and evaluates how well they **transfer** to robust target models.  Results are compared against each target's **reported AutoAttack robust accuracy** to analyze transferability.

### Models

| Role | Model | Architecture | Source |
|------|-------|-------------|--------|
| Surrogate | `Standard` | WideResNet-28-10 | RobustBench (non-robust baseline) |
| Surrogate | `cifar10_vgg16_bn` | VGG-16-BN | chenyaofo/pytorch-cifar-models |
| Surrogate | `cifar10_resnet56` | Resnet-56 | chenyaofo/pytorch-cifar-models |
| Surrogate | `cifar10_mobilenetv2_x1_4` | MobileNetV2 | chenyaofo/pytorch-cifar-models |
| Surrogate | `cifar10_repvgg_a2` | RepVGG | chenyaofo/pytorch-cifar-models |
| Target | `Bartoldson2024Adversarial_WRN-94-16` | WideResNet-94-16 | RobustBench (73.71% robust acc) |
| Target | `Ding2020MMA` | WideResNet-28-4	 | RobustBench (41.44% robust acc) |
| Target | `Bai2024MixedNUTS` | ResNet-152 + WideResNet-70-16 | RobustBench (69.71% robust acc) |

### Attacks (from `torchattack`)

| Attack | Description |
|--------|-------------|
| DI-FGSM | Diverse Input FGSM (input diversity) |
| TI-FGSM | Translation-Invariant FGSM |
| Admix | Admix input transformation |
| VMI-FGSM | Variance-tuned MI-FGSM |

## Setup

### Prerequisites

- Python 3.10+ with PyTorch, in a conda environment.
- Minimum 16 GB NVIDIA GPU recommended.
- \>=8 GB RAM recommended.

### Install dependencies

```bash
pip install -r requirements.txt
```

All dependencies (torch, torchvision, robustbench, torchattack, etc.) should already be installed if you followed the project setup. Install suitable CUDA version of PyTorch for your device.

## Usage

### Quick test (100 samples)

```bash
python main.py --num-samples 100
```

### Full run (1000 samples, default config)

```bash
python main.py
```

### Full dataset (10,000 samples)

```bash
python main.py --num-samples 10000
```

### Skip ensemble attacks

```bash
python main.py --no-ensemble
```

### Custom config

```bash
python main.py --config configs/my_config.yaml
```

### Force CPU

```bash
python main.py --device cpu
```

## Output

Results are saved to `results/`:

| File | Description |
|------|-------------|
| `results.csv` | Full results table |
| `summary.md` | Markdown-formatted summary table |
| `attack_success_rates.png` | Bar chart of attack success rates by attacks and target |
| `best_transfer_vs_reported.png` | Best transfer result vs reported AutoAttack for each target |
| `config_snapshot.yaml` | Copy of the config used for the run |

## Project Structure

```
transfer_attack_study/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml           # Default experiment configuration
├── src/
│   ├── __init__.py
│   ├── data.py                # CIFAR-10 data loading
│   ├── models.py              # Model loading and normalization wrappers
│   ├── attacks.py             # Attack factory and adversarial generation
│   ├── evaluate.py            # Transfer evaluation metrics
│   ├── metadata.py            # Reported model accuracies
│   └── reporting.py           # CSV, markdown, and plot generation
├── main.py         # Main experiment runner
└── results/                   # Output directory (created at runtime)
```

## Citations
- [RobustBench](https://arxiv.org/abs/2010.09670): Croce et al., 2021
- [torchattack](https://github.com/spencerwooo/torchattack): SpencerWoo
- [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models)

