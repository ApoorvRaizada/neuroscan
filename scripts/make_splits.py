"""Build leakage-aware train/val/test splits from near-duplicate clusters."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
MAX_DIST = 2  # pHash distance <= this => same cluster
FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
ROOT = Path("data/raw")
OUT = Path("data/splits")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
EXT = {".jpg", ".jpeg", ".png"}


def posix(p):
    return Path(p).as_posix()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows = []
    for split in ("Training", "Testing"):
        for cls in CLASSES:
            for p in sorted((ROOT / split / cls).iterdir()):
                if p.suffix.lower() in EXT:
                    rows.append({"path": posix(p), "label": cls, "orig_split": split})
    df = pd.DataFrame(rows)

    # Union-find over near-duplicate pairs
    parent = {p: p for p in df.path}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = pd.read_csv("reports/near_duplicate_pairs.csv")
    pairs = pairs[pairs.dist <= MAX_DIST]
    for a, b in zip(pairs.path_a, pairs.path_b):
        ra, rb = find(posix(a)), find(posix(b))
        if ra != rb:
            parent[rb] = ra
    df["cluster"] = df.path.map(find)

    # Drop clusters with conflicting labels
    n_labels = df.groupby("cluster").label.nunique()
    bad = n_labels[n_labels > 1].index
    dropped = int(df.cluster.isin(bad).sum())
    df = df[~df.cluster.isin(bad)].copy()

    # Assign whole clusters to splits, stratified by label (greedy by deficit)
    clusters = df.groupby("cluster").agg(label=("label", "first"), size=("path", "size")).reset_index()
    assign = {}
    for cls in CLASSES:
        sub = clusters[clusters.label == cls].sample(frac=1, random_state=SEED)
        total = sub["size"].sum()
        target = {s: f * total for s, f in FRACS.items()}
        have = {s: 0 for s in FRACS}
        for _, r in sub.iterrows():
            s = max(FRACS, key=lambda k: (target[k] - have[k]) / target[k])
            assign[r.cluster] = s
            have[s] += r["size"]
    df["split"] = df.cluster.map(assign)

    # val/test: one image per cluster; train keeps all
    keep = df[df.split == "train"]
    for s in ("val", "test"):
        keep = pd.concat([keep, df[df.split == s].drop_duplicates("cluster")])

    # Sanity check: no cluster appears in more than one split
    cs = {s: set(keep[keep.split == s].cluster) for s in FRACS}
    assert not (cs["train"] & cs["val"] or cs["train"] & cs["test"] or cs["val"] & cs["test"])

    summary = {"seed": SEED, "max_dist": MAX_DIST, "total_images": len(rows),
               "images_dropped_label_conflict": dropped,
               "clusters": int(df.cluster.nunique()), "splits": {}}
    for s in FRACS:
        part = keep[keep.split == s][["path", "label", "cluster", "orig_split"]]
        part.to_csv(OUT / f"{s}.csv", index=False)
        summary["splits"][s] = {"images": len(part),
                                "per_class": part.label.value_counts().to_dict()}
    (OUT / "splits_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()