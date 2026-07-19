# Plots training curves and evaluation results

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_training_curves(results_csv: str, save_dir: str = "outputs"):
    df = pd.read_csv(results_csv)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Results", fontsize=14)

    axes[0,0].plot(df["epoch"], df["train/box_loss"], label="Train")
    axes[0,0].set_title("Box Loss")

    axes[0,1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50", color="green")
    axes[0,1].set_title("mAP50")

    axes[1,0].plot(df["epoch"], df["metrics/precision(B)"], label="Precision", color="blue")
    axes[1,0].set_title("Precision")

    axes[1,1].plot(df["epoch"], df["metrics/recall(B)"], label="Recall", color="orange")
    axes[1,1].set_title("Recall")

    for ax in axes.flat:
        ax.legend()
        ax.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.show()
    print(f"Saved to {save_dir}/training_curves.png")

def plot_ood_comparison(csv_path: str = "outputs/metrics/ood_summary.csv"):
    df = pd.read_csv(csv_path, index_col=0)
    ax = df[["mAP50", "mAP50-95"]].plot(kind="bar", figsize=(10, 5), rot=30)
    ax.set_title("OOD Evaluation — mAP Comparison")
    ax.set_ylabel("Score")
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig("outputs/metrics/ood_comparison.png")
    plt.show()