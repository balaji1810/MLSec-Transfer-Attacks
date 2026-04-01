import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import torch


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results)


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"CSV saved to {path}")


def attack_column_prefix(attack_name: str) -> str:
    prefix = "".join(
        ch.lower() if ch.isalnum() else "_"
        for ch in str(attack_name)
    )
    while "__" in prefix:
        prefix = prefix.replace("__", "_")
    return prefix.strip("_")


def build_wide_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"surrogate", "target", "attack"}
    if not required_cols.issubset(df.columns):
        return df.copy()

    key_cols = ["surrogate", "target"]
    base_cols = [
        c for c in ["clean_accuracy", "reported_robust_acc"]
        if c in df.columns
    ]

    wide_df = (
        df[key_cols + base_cols]
        .groupby(key_cols, as_index=False)
        .first()
        .sort_values(key_cols)
        .reset_index(drop=True)
    )

    attacks = sorted(df["attack"].dropna().unique(), key=lambda x: str(x).lower())
    for attack_name in attacks:
        attack_rows = df[df["attack"] == attack_name]
        metric_cols = []
        rename_map = {}

        if "adversarial_accuracy" in attack_rows.columns:
            metric_cols.append("adversarial_accuracy")
            rename_map["adversarial_accuracy"] = (
                f"{attack_column_prefix(attack_name)}_adversarial_acc"
            )
        if "attack_success_rate" in attack_rows.columns:
            metric_cols.append("attack_success_rate")
            rename_map["attack_success_rate"] = (
                f"{attack_column_prefix(attack_name)}_asr"
            )

        if not metric_cols:
            continue

        per_attack = (
            attack_rows[key_cols + metric_cols]
            .groupby(key_cols, as_index=False)
            .first()
            .rename(columns=rename_map)
        )
        wide_df = wide_df.merge(per_attack, on=key_cols, how="left")

    return wide_df


def print_summary_table(df: pd.DataFrame) -> None:
    summary_df = build_wide_summary_dataframe(df)
    fixed_cols = ["surrogate", "target", "clean_accuracy", "reported_robust_acc"]
    attack_cols = [
        c for c in summary_df.columns
        if c.endswith("_adversarial_acc") or c.endswith("_asr")
    ]
    display_cols = [c for c in fixed_cols + attack_cols if c in summary_df.columns]
    if not display_cols:
        display_cols = list(summary_df.columns)

    print(tabulate(summary_df[display_cols], headers="keys", # pyright: ignore[reportArgumentType]
                   tablefmt="grid", showindex=False, floatfmt=".2f"))


def save_summary_markdown(df: pd.DataFrame, path: str) -> None:
    summary_df = build_wide_summary_dataframe(df)
    fixed_cols = ["surrogate", "target", "clean_accuracy", "reported_robust_acc"]
    attack_cols = [
        c for c in summary_df.columns
        if c.endswith("_adversarial_acc") or c.endswith("_asr")
    ]
    display_cols = [c for c in fixed_cols + attack_cols if c in summary_df.columns]
    if not display_cols:
        display_cols = list(summary_df.columns)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# Transfer Attack Results\n\n")
        f.write(tabulate(summary_df[display_cols], headers="keys", # pyright: ignore[reportArgumentType]
                         tablefmt="github", showindex=False, floatfmt=".2f"))
        f.write("\n")
    print(f"Markdown summary saved to {path}")


def plot_transfer_accuracy_by_attack(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df.pivot_table(
        index="attack", columns="target",
        values="attack_success_rate", aggfunc="mean",
    )
    pivot.plot(kind="bar", ax=ax, edgecolor="black", width=0.8)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_xlabel("Attack Method")
    ax.set_title("Attack Success Rate by Attack and Target")
    ax.legend(title="Target Model", loc="upper left")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Plot saved to {path}")


def plot_transfer_attack_vs_reported(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    best = df.groupby("target").agg(
        transfer_adv_acc=("adversarial_accuracy", "min"),
        reported_robust_acc=("reported_robust_acc", "first"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(best))
    width = 0.35
    bars1 = ax.bar(
        [i - width / 2 for i in x], best["reported_robust_acc"],
        width, label="Reported AutoAttack Robust Acc", color="#4C72B0",
        edgecolor="black",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x], best["transfer_adv_acc"],
        width, label="Transfer Adv Acc", color="#DD8452",
        edgecolor="black",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(best["target"], rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Transfer Attack vs Reported AutoAttack Robustness")
    ax.legend()
    ax.set_ylim(0, 100)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Plot saved to {path}")


def generate_all_reports(results: list[dict], output_dir: str) -> pd.DataFrame:
    df = results_to_dataframe(results)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70 + "\n")
    print_summary_table(df)

    # CSV
    # save_csv(df, os.path.join(output_dir, f"results_{time.time()}.csv"))
    save_csv(df, os.path.join(output_dir, f"results.csv"))

    # Markdown
    # save_summary_markdown(df, os.path.join(output_dir, f"summary_{time.time()}.md"))
    save_summary_markdown(df, os.path.join(output_dir, f"summary.md"))

    # Plots
    # plot_transfer_accuracy_by_attack(
    #     df, os.path.join(output_dir, f"transfer_accuracy_by_attack_{time.time()}.png")
    # )
    plot_transfer_accuracy_by_attack(
        df, os.path.join(output_dir, f"attack_success_rates.png")
    )
    # plot_transfer_attack_vs_reported(
    #     df, os.path.join(output_dir, f"transfer_attack_vs_reported_{time.time()}.png")
    # )
    plot_transfer_attack_vs_reported(
        df, os.path.join(output_dir, f"transfer_attack_vs_reported.png")
    )

    return df

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

def save_fooled_images(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    clean_preds: torch.Tensor,
    adv_preds: torch.Tensor,
    surrogate_name: str,
    attack_name: str,
    target_name: str,
    output_dir: str,
) -> None:
    save_dir = os.path.join(output_dir, f"{surrogate_name}__{attack_name}__{target_name}")
    os.makedirs(save_dir, exist_ok=True)

    n = len(labels)

    # Save raw tensors
    torch.save({
        "clean_images": clean_images,
        "adv_images": adv_images,
        "labels": labels,
        "clean_preds": clean_preds,
        "adv_preds": adv_preds,
    }, os.path.join(save_dir, "fooled_data.pt"))

    for i in range(n):
        true_label = CIFAR10_CLASSES[labels[i]]
        pred_clean = CIFAR10_CLASSES[clean_preds[i]]
        pred_adv = CIFAR10_CLASSES[adv_preds[i]]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        ax1, ax2 = axes

        ax1.imshow(clean_images[i].permute(1, 2, 0).cpu())
        ax1.set_title("Original", fontsize=10)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel(f"True: {true_label}\nClean pred: {pred_clean}", fontsize=9)

        ax2.imshow(adv_images[i].permute(1, 2, 0).cpu())
        ax2.set_title("Adversarial", fontsize=10)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel(f"Adv pred: {pred_adv}", fontsize=9)

        fig.tight_layout(pad=0.5)

        filename = f"{i:04d}_true_{true_label}_clean_{pred_clean}_adv_{pred_adv}.png"
        filename = filename.replace(" ", "_")
        out_path = os.path.join(save_dir, filename)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {n} fooled image pairs to {save_dir}/")
