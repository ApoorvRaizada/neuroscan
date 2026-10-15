"""Inspect the dataset: class balance, image sizes/modes, sample grid."""
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("data/raw")
OUT = Path("reports/figures")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
EXT = {".jpg", ".jpeg", ".png"}
random.seed(42)


def list_files(split, cls):
    return sorted(p for p in (ROOT / split / cls).iterdir() if p.suffix.lower() in EXT)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Class balance
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, split in zip(axes, ("Training", "Testing")):
        counts = [len(list_files(split, c)) for c in CLASSES]
        ax.bar(CLASSES, counts)
        ax.set_title(f"{split} class balance")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT / "class_balance.png", dpi=120)

    # 2. Image sizes and color modes (reads headers only, fast)
    sizes, modes = Counter(), Counter()
    for split in ("Training", "Testing"):
        for c in CLASSES:
            for p in list_files(split, c):
                with Image.open(p) as im:
                    sizes[im.size] += 1
                    modes[im.mode] += 1
    print("Unique image sizes:", len(sizes))
    print("Most common sizes (w, h): count")
    for s, n in sizes.most_common(8):
        print(f"  {s}: {n}")
    print("Color modes:", dict(modes))

    # 3. Sample grid: 4 random training images per class
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for r, c in enumerate(CLASSES):
        picks = random.sample(list_files("Training", c), 4)
        for k, p in enumerate(picks):
            axes[r, k].imshow(Image.open(p).convert("L"), cmap="gray")
            axes[r, k].axis("off")
            if k == 0:
                axes[r, k].set_title(c, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "sample_grid.png", dpi=120)
    print("Saved figures to", OUT)


if __name__ == "__main__":
    main()