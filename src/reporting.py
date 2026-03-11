import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results)


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"CSV saved to {path}")


def print_summary_table(df: pd.DataFrame) -> None:
    cols = [
        "surrogate", "attack", "target",
        "clean_accuracy", "adversarial_accuracy",
        "attack_success_rate",
        # "reported_robust_acc",
        # "delta_vs_reported",
    ]
    display_cols = [c for c in cols if c in df.columns]
    print(tabulate(df[display_cols], headers="keys", tablefmt="grid", # pyright: ignore[reportArgumentType]
                   showindex=False, floatfmt=".2f"))


def save_summary_markdown(df: pd.DataFrame, path: str) -> None:
    cols = [
        "surrogate", "attack", "target",
        "clean_accuracy", "adversarial_accuracy",
        "attack_success_rate", "reported_robust_acc",
        "delta_vs_reported",
    ]
    display_cols = [c for c in cols if c in df.columns]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# Transfer Attack Results\n\n")
        f.write(tabulate(df[display_cols], headers="keys", # pyright: ignore[reportArgumentType]
                         tablefmt="github", showindex=False, floatfmt=".2f"))
        f.write("\n")
    print(f"Markdown summary saved to {path}")


def plot_transfer_accuracy_by_attack(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df.pivot_table(
        index="attack", columns="target",
        values="adversarial_accuracy", aggfunc="mean",
    )
    pivot.plot(kind="bar", ax=ax, edgecolor="black", width=0.8)
    ax.set_ylabel("Transfer Adversarial Accuracy (%)")
    ax.set_xlabel("Attack Method")
    ax.set_title("Transfer Adversarial Accuracy by Attack and Target")
    ax.legend(title="Target Model", loc="upper left")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Plot saved to {path}")


def plot_best_transfer_vs_reported(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    best = df.groupby("target").agg(
        best_transfer_adv_acc=("adversarial_accuracy", "min"),
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
        [i + width / 2 for i in x], best["best_transfer_adv_acc"],
        width, label="Best Transfer Adv Acc", color="#DD8452",
        edgecolor="black",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(best["target"], rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Best Transfer Attack vs Reported AutoAttack Robustness")
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
    save_csv(df, os.path.join(output_dir, f"results_{time.time()}.csv"))

    # Markdown
    save_summary_markdown(df, os.path.join(output_dir, f"summary_{time.time()}.md"))

    # Plots
    plot_transfer_accuracy_by_attack(
        df, os.path.join(output_dir, f"transfer_accuracy_by_attack_{time.time()}.png")
    )
    plot_best_transfer_vs_reported(
        df, os.path.join(output_dir, f"best_transfer_vs_reported_{time.time()}.png")
    )

    return df
