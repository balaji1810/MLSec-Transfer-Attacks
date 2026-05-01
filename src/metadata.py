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
    "Bartoldson2024Adversarial_WRN-94-16": {
        "reported_clean_acc": 93.68,
        "reported_robust_acc": 73.71,
    },
    "Ding2020MMA": {
        "reported_clean_acc": 84.36,
        "reported_robust_acc": 41.44,
    },
    "Bai2024MixedNUTS" : {
        "reported_clean_acc": 95.19,
        "reported_robust_acc": 69.71,
    },
}


def get_reported_robust_acc(model_name: str) -> float | None:
    meta = model_metadata.get(model_name, {})
    return meta.get("reported_robust_acc")


def get_reported_clean_acc(model_name: str) -> float | None:
    meta = model_metadata.get(model_name, {})
    return meta.get("reported_clean_acc")
