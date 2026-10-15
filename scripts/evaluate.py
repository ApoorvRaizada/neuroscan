"""Evaluate trained models on the held-out test split (leakage-aware)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from torch.utils.data import DataLoader

from model.dataset import CLASSES, MRIDataset
from model.networks import BaselineCNN, build_resnet18


def predict(net, loader, device):
    net.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.softmax(net(x.to(device)), 1).cpu().numpy())
            ys += y.tolist()
    return np.concatenate(probs), np.array(ys)


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), CLASSES, rotation=30)
    ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["baseline", "resnet18"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = MRIDataset("test", train=False)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # Majority-class reference: always predict the most common TRAIN class
    train_labels = pd.read_csv("data/splits/train.csv").label
    majority = CLASSES.index(train_labels.value_counts().idxmax())
    y_true = np.array([CLASSES.index(l) for l in test_ds.df.label])
    maj_pred = np.full_like(y_true, majority)
    reference = {"predicts": CLASSES[majority],
                 "accuracy": float(accuracy_score(y_true, maj_pred)),
                 "macro_f1": float(f1_score(y_true, maj_pred, average="macro", zero_division=0))}
    print("Majority-class reference:", reference)

    for name in args.models:
        net = BaselineCNN() if name == "baseline" else build_resnet18(pretrained=False)
        net.load_state_dict(torch.load(f"runs/{name}/best.pt", map_location=device))
        net.to(device)
        probs, y = predict(net, loader, device)
        pred = probs.argmax(1)

        auc = {c: float(roc_auc_score(y == i, probs[:, i])) for i, c in enumerate(CLASSES)}
        report = classification_report(y, pred, target_names=CLASSES, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, pred, labels=range(4))
        result = {"model": name, "test_images": int(len(y)),
                  "accuracy": float(accuracy_score(y, pred)),
                  "macro_f1": float(f1_score(y, pred, average="macro")),
                  "macro_auc_ovr": float(np.mean(list(auc.values()))),
                  "auc_per_class": auc, "per_class": {c: report[c] for c in CLASSES},
                  "confusion_matrix": cm.tolist(), "majority_reference": reference}
        Path(f"runs/{name}/test_metrics.json").write_text(json.dumps(result, indent=2))
        plot_cm(cm, f"{name} (test)", f"reports/figures/cm_{name}.png")

        print(f"\n=== {name} | {len(y)} test images ===")
        print(classification_report(y, pred, target_names=CLASSES, digits=3, zero_division=0))
        print("ROC-AUC (one-vs-rest):", {k: round(v, 4) for k, v in auc.items()})
        print(f"accuracy {result['accuracy']:.4f} | macro-F1 {result['macro_f1']:.4f} "
              f"| macro-AUC {result['macro_auc_ovr']:.4f}")


if __name__ == "__main__":
    main()