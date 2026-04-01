import argparse
import os
import time
import json

import torch
import numpy as np
import yaml

from src.data import load_cifar10_testset
from src.models import ModelWrapper, load_surrogate, load_target, build_ensemble
from src.attacks import create_attack, generate_adversarial_examples
from src.evaluate import evaluate_transfer
from src.metadata import get_reported_robust_acc, get_reported_clean_acc
from src.reporting import generate_all_reports, save_fooled_images


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def run_single_experiment(
    surrogate_name: str,
    surrogate_model: torch.nn.Module,
    normalize_fn,
    attack_name: str,
    attack_params: dict,
    target_configs: list[dict],
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    batch_size: int,
    model_dir: str,
    device: torch.device,
    output_dir: str,
    save_fooled: bool = True,
) -> list[dict]:
    print(f"\n{'-' * 60}")
    print(f"Surrogate: {surrogate_name}  |  Attack: {attack_name}")
    print(f"{'-' * 60}")

    # Create attack
    attack = create_attack(
        attack_name=attack_name,
        model=surrogate_model,
        normalize=normalize_fn,
        device=device,
        eps=eps,
        **attack_params,
    )

    # Generate adversarial examples on surrogate
    t0 = time.time()
    adv_images = generate_adversarial_examples(
        attack=attack,
        images=images,
        labels=labels,
        batch_size=batch_size,
        device=device,
    )
    gen_time = time.time() - t0
    print(f"Adversarial examples generated in {gen_time:.1f}s")

    # Evaluate transfer to each target
    results = []
    for tgt_cfg in target_configs:
        tgt_name = tgt_cfg["name"]
        print(f"Evaluating transfer to {tgt_name}")

        target_model = load_target(tgt_cfg, model_dir=model_dir, device=device)

        metrics = evaluate_transfer(
            target_model=target_model,
            clean_images=images,
            adv_images=adv_images,
            labels=labels,
            batch_size=batch_size,
            device=device,
        )

        # Save fooled adversarial images
        fooled_mask = metrics.pop("fooled_mask")
        if fooled_mask.any() and save_fooled:
            adv_preds = metrics.pop("adv_preds")
            clean_preds = metrics.pop("clean_preds")
            save_fooled_images(
                clean_images=images[fooled_mask],
                adv_images=adv_images[fooled_mask],
                labels=labels[fooled_mask],
                clean_preds=clean_preds[fooled_mask],
                adv_preds=adv_preds[fooled_mask],
                surrogate_name=surrogate_name,
                attack_name=attack_name,
                target_name=tgt_name,
                output_dir=os.path.join(output_dir, "fooled_images"),
            )

        # Reported values
        reported_robust = get_reported_robust_acc(tgt_name)
        reported_clean = get_reported_clean_acc(tgt_name)

        # Delta: how much lower our transfer attack acc is vs reported robust acc
        # Negative delta means our attack is "stronger" than reported AutoAttack
        delta = None
        if reported_robust is not None:
            delta = round(metrics["adversarial_accuracy"] - reported_robust, 2)

        result = {
            "surrogate": surrogate_name,
            "attack": attack_name,
            "target": tgt_name,
            "num_samples": metrics["num_samples"],
            "clean_accuracy": metrics["clean_accuracy"],
            "adversarial_accuracy": metrics["adversarial_accuracy"],
            "attack_success_rate": metrics["attack_success_rate"],
            "mean_linf_perturbation": metrics["mean_linf_perturbation"],
            "max_linf_perturbation": metrics["max_linf_perturbation"],
            "reported_clean_acc": reported_clean,
            "reported_robust_acc": reported_robust,
            "delta_vs_reported": delta,
            "generation_time_s": round(gen_time, 1),
        }
        results.append(result)

        print(f"Clean acc: {metrics['clean_accuracy']:.2f}%  "
              f"Adv acc: {metrics['adversarial_accuracy']:.2f}%  "
              f"ASR: {metrics['attack_success_rate']:.2f}%  "
              f"Linf: {metrics['mean_linf_perturbation']:.4f}")

        # Free target model to save VRAM
        del target_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Override number of test samples (default: from config)",
    )
    # parser.add_argument(
    #     "--no-ensemble", action="store_true",
    #     help="Skip ensemble surrogate attacks",
    # )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cuda/cpu)",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Overrides
    if args.num_samples is not None:
        cfg["data"]["num_samples"] = args.num_samples
    if args.device is not None:
        cfg["device"] = args.device
    # if args.no_ensemble:
    #     cfg["ensemble"]["enabled"] = False

    # Setup
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    eps = cfg["threat_model"]["epsilon"]
    batch_size = cfg["data"]["batch_size"]
    model_dir = cfg.get("model_dir", "./models")
    output_dir = cfg["output"]["results_dir"]

    print("=" * 70)
    print("Transfer-Based Adversarial Attack on CIFAR-10")
    print("=" * 70)
    print(f"Device:       {device}")
    print(f"Seed:         {seed}")
    print(f"Epsilon:      {eps} ({eps * 255:.1f}/255)")
    print(f"Num samples:  {cfg['data']['num_samples']}")
    print(f"Surrogates:   {[s['name'] for s in cfg['surrogates']]}")
    print(f"Targets:      {[t['name'] for t in cfg['targets']]}")
    print(f"Attacks:      {[a['name'] for a in cfg['attacks']]}")
    print(f"Ensemble:     {cfg['ensemble']['enabled']}")
    print()

    # Load data
    print("Loading CIFAR-10 test data")
    images, labels = load_cifar10_testset(
        data_dir=cfg["data"]["data_dir"],
        num_samples=cfg["data"]["num_samples"],
        seed=seed,
    )
    print(f"Loaded {len(labels)} samples, shape={images.shape}")

    all_results = []

    # Single-surrogate attacks
    if cfg["enable_single_surrogate_attacks"]:
        for surr_cfg in cfg["surrogates"]:
            surr_name = surr_cfg["name"]
            print(f"\n{'=' * 70}")
            print(f"Loading surrogate: {surr_name}")
            print(f"{'=' * 70}")

            surrogate_model, normalize_fn = load_surrogate(
                surr_cfg, model_dir=model_dir, device=device
            )

            surrogate_model = ModelWrapper(surrogate_model, normalize_fn).to(device)

            for atk_cfg in cfg["attacks"]:
                atk_name = atk_cfg["name"]
                atk_params = dict(atk_cfg.get("params", {}))

                results = run_single_experiment(
                    surrogate_name=surr_name,
                    surrogate_model=surrogate_model,
                    normalize_fn=None,
                    attack_name=atk_name,
                    attack_params=atk_params,
                    target_configs=cfg["targets"],
                    images=images,
                    labels=labels,
                    eps=eps,
                    batch_size=batch_size,
                    model_dir=model_dir,
                    device=device,
                    output_dir=output_dir,
                    save_fooled=cfg["output"]["save_adv_examples"],
                )
                all_results.extend(results)

            # Free surrogate to save VRAM
            del surrogate_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Ensemble surrogate attacks
    if cfg["ensemble"]["enabled"]:
        print(f"\n{'=' * 70}")
        print("Loading ensemble surrogate (all surrogates combined)")
        print(f"{'=' * 70}")

        ensemble_model, ensemble_norm_fn = build_ensemble(
            cfg["surrogates"], model_dir=model_dir, device=device
        )

        for atk_cfg in cfg["attacks"]:
            atk_name = atk_cfg["name"]
            atk_params = dict(atk_cfg.get("params", {}))

            results = run_single_experiment(
                surrogate_name="Ensemble",
                surrogate_model=ensemble_model,
                normalize_fn=ensemble_norm_fn,
                attack_name=atk_name,
                attack_params=atk_params,
                target_configs=cfg["targets"],
                images=images,
                labels=labels,
                eps=eps,
                batch_size=batch_size,
                model_dir=model_dir,
                device=device,
                output_dir=output_dir,
                save_fooled=cfg["output"]["save_adv_examples"],
            )
            all_results.extend(results)

        del ensemble_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final reports
    df = generate_all_reports(all_results, output_dir)

    # Save config snapshot
    config_snapshot_path = os.path.join(output_dir, f"config_snapshot_{seed}_{time.time()}.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_snapshot_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\n{'=' * 70}")
    print("Experiment complete")
    print(f"Results saved to: {output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
