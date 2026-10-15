"""Visually check near-duplicate pairs across Train/Test, grouped by hash distance."""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

random.seed(42)
OUT = Path("reports/figures")
df = pd.read_csv("reports/near_duplicate_pairs.csv")
cross = df[df.split_a != df.split_b].copy()
cross["disagree"] = cross.label_a != cross.label_b

print("Cross-split pairs by pHash distance:")
print(cross.groupby("dist").disagree.agg(pairs="size", label_disagree="sum"))

for lo, hi in [(0, 1), (2, 3), (4, 6)]:
    sub = cross[(cross.dist >= lo) & (cross.dist <= hi)]
    if sub.empty:
        print(f"No pairs with distance {lo}-{hi}")
        continue
    sub = sub.sample(min(5, len(sub)), random_state=42)
    fig, axes = plt.subplots(len(sub), 2, figsize=(5, 2.4 * len(sub)), squeeze=False)
    for r, (_, row) in enumerate(sub.iterrows()):
        for c, side in enumerate(("a", "b")):
            axes[r, c].imshow(Image.open(row[f"path_{side}"]).convert("L"), cmap="gray")
            axes[r, c].set_title(f"{row[f'split_{side}']} / {row[f'label_{side}']} (d={row.dist})", fontsize=7)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"pairs_dist_{lo}_{hi}.png", dpi=110)
    print("Saved", OUT / f"pairs_dist_{lo}_{hi}.png")