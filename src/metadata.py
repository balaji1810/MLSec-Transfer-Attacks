model_metadata: dict[str, dict] = {
    # Surrogate (non-robust)
    "Standard": {
        "reported_clean_acc": 94.78,
        "reported_robust_acc": 0.0,
    },
    "cifar10_vgg16_bn": {
        "reported_clean_acc": 93.96,
        "reported_robust_acc": None,
    },
    "resnet56": {
        "reported_clean_acc": 94.37,
        "reported_robust_acc": None,
    },
    "vgg16_bn" : {
        "reported_clean_acc": 94.16,
        "reported_robust_acc": None,
    },
    "mobilenetv2_x1_4": {
        "reported_clean_acc": 94.22,
        "reported_robust_acc": None,
    },
    "shufflenetv2_x2_0": {
        "reported_clean_acc": 93.81,
        "reported_robust_acc": None,
    },
    "repvgg_a0": {
        "reported_clean_acc": 94.39,
        "reported_robust_acc": None,
    },
    # Targets (robust)
    "Carmon2019Unlabeled": {
        "reported_clean_acc": 89.69,
        "reported_robust_acc": 59.53,
    },
    "Rebuffi2021Fixing_70_16_cutmix_extra": {
        "reported_clean_acc": 92.23,
        "reported_robust_acc": 66.56,
    },
    "Wang2023Better_WRN-70-16": {
        "reported_clean_acc": 93.25,
        "reported_robust_acc": 70.69,
    },
    "Chen2020Adversarial": {
        "reported_clean_acc": 86.04,
        "reported_robust_acc": 51.56,
    },
    "Kang2021Stable": {
        "reported_clean_acc": 93.73,
        "reported_robust_acc": 64.20,
    },
    "Bartoldson2024Adversarial_WRN-94-16": {
        "reported_clean_acc": 93.68,
        "reported_robust_acc": 73.71,
    },
    "Gowal2021Improving_28_10_ddpm_100m": {
        "reported_clean_acc": 87.50,
        "reported_robust_acc": 63.38,
    },
    "Sehwag2021Proxy_R18": {
        "reported_clean_acc": 84.59,
        "reported_robust_acc": 55.54,
    },
    "Addepalli2022Efficient_RN18": {
        "reported_clean_acc": 86.04,
        "reported_robust_acc": 51.56,
    },
    "Xu2023Exploring_WRN-28-10": {
        "reported_clean_acc": 93.69,
        "reported_robust_acc": 63.89,
    },
    "Rice2020Overfitting": {
        "reported_clean_acc": 85.34,
        "reported_robust_acc": 53.42,
    },
    "Wong2020Fast": {
        "reported_clean_acc": 83.34,
        "reported_robust_acc": 43.21,
    },
    "Andriushchenko2020Understanding": {
        "reported_clean_acc": 79.84,
        "reported_robust_acc": 43.93,
    },
    "Ding2020MMA": {
        "reported_clean_acc": 84.36,
        "reported_robust_acc": 41.44,
    },
    "Engstrom2019Robustness": {
        "reported_clean_acc": 87.03,
        "reported_robust_acc": 49.25,
    },
    "Zhang2019You": {
        "reported_clean_acc": 87.20,
        "reported_robust_acc": 44.83,
    },
    "Sitawarin2020Improving": {
        "reported_clean_acc": 86.84,
        "reported_robust_acc": 43.21,
    },
    "Huang2020Self" : {
        "reported_clean_acc": 83.48,
        "reported_robust_acc": 53.34,
    },
    "Chen2020Efficient" : {
        "reported_clean_acc": 85.32,
        "reported_robust_acc": 51.12,
    },
    "Chen2020Adversarial" : {
        "reported_clean_acc": 86.04,
        "reported_robust_acc": 51.56,
    },
}


def get_reported_robust_acc(model_name: str) -> float | None:
    meta = model_metadata.get(model_name, {})
    return meta.get("reported_robust_acc")


def get_reported_clean_acc(model_name: str) -> float | None:
    meta = model_metadata.get(model_name, {})
    return meta.get("reported_clean_acc")