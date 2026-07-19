import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Find all results.csv files
csv_files = glob.glob("runs/**/results.csv", recursive=True)
print("Found results files:")
for f in csv_files:
    print("  " + f)

if not csv_files:
    print("No results.csv found!")
    exit()

# Load all results
all_results = {}
for csv_path in csv_files:
    name = csv_path.split(os.sep)[-2]
    df   = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    all_results[name] = df
    print("Loaded: " + name + " (" + str(len(df)) + " epochs)")

# Plot each model
for name, df in all_results.items():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("PatchFinders - " + name, fontsize=14)

    axes[0,0].plot(df["epoch"], df["train/box_loss"], label="Train", color="blue")
    axes[0,0].plot(df["epoch"], df["val/box_loss"],   label="Val",   color="orange")
    axes[0,0].set_title("Box Loss")
    axes[0,0].legend()
    axes[0,0].grid(True)

    axes[0,1].plot(df["epoch"], df["train/cls_loss"], label="Train", color="blue")
    axes[0,1].plot(df["epoch"], df["val/cls_loss"],   label="Val",   color="orange")
    axes[0,1].set_title("Class Loss")
    axes[0,1].legend()
    axes[0,1].grid(True)

    axes[0,2].plot(df["epoch"], df["train/dfl_loss"], label="Train", color="blue")
    axes[0,2].plot(df["epoch"], df["val/dfl_loss"],   label="Val",   color="orange")
    axes[0,2].set_title("DFL Loss")
    axes[0,2].legend()
    axes[0,2].grid(True)

    axes[1,0].plot(df["epoch"], df["metrics/mAP50(B)"], color="green")
    axes[1,0].set_title("mAP50")
    axes[1,0].grid(True)

    axes[1,1].plot(df["epoch"], df["metrics/precision(B)"], color="purple")
    axes[1,1].set_title("Precision")
    axes[1,1].grid(True)

    axes[1,2].plot(df["epoch"], df["metrics/recall(B)"], color="brown")
    axes[1,2].set_title("Recall")
    axes[1,2].grid(True)

    plt.tight_layout()
    os.makedirs("outputs/metrics", exist_ok=True)
    save_path = "outputs/metrics/" + name + "_curves.png"
    plt.savefig(save_path, dpi=150)
    print("Saved: " + save_path)
    plt.show()

# Compare all models on mAP50
if len(all_results) > 1:
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["green", "blue", "red", "orange"]
    for i, (name, df) in enumerate(all_results.items()):
        if "metrics/mAP50(B)" in df.columns:
            ax.plot(df["epoch"], df["metrics/mAP50(B)"],
                    label=name, color=colors[i % len(colors)])

    ax.set_title("mAP50 Comparison — All Models")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/metrics/model_comparison.png", dpi=150)
    print("Saved: outputs/metrics/model_comparison.png")
    plt.show()