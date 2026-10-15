"""Find exact (SHA-256) and near-duplicate (perceptual hash) images."""
import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data/raw")
OUT = Path("reports")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
EXT = {".jpg", ".jpeg", ".png"}
NEAR_THRESHOLD = 6  # max differing bits (out of 64) to call "near-duplicate"


def main():
    rows, bits = [], []
    for split in ("Training", "Testing"):
        for cls in CLASSES:
            for p in sorted((ROOT / split / cls).iterdir()):
                if p.suffix.lower() not in EXT:
                    continue
                sha = hashlib.sha256(p.read_bytes()).hexdigest()
                with Image.open(p) as im:
                    h = imagehash.phash(im.convert("L"))
                rows.append({"path": str(p), "split": split, "label": cls, "sha": sha})
                bits.append(h.hash.flatten())
    df = pd.DataFrame(rows)
    B = np.array(bits, dtype=np.float32)
    n = len(df)
    print("Images hashed:", n)

    # Exact duplicates
    dup_groups = df.groupby("sha").filter(lambda g: len(g) > 1)
    train_sha = set(df[df.split == "Training"].sha)
    test_in_train = df[(df.split == "Testing") & df.sha.isin(train_sha)]
    print("\n== Exact duplicates (identical bytes) ==")
    print("Images involved in any exact-duplicate group:", len(dup_groups))
    print("Test images with an exact copy in Training:", len(test_in_train))

    # Near duplicates via perceptual hash (chunked to save memory)
    pairs = []
    for i in range(0, n, 500):
        A = B[i:i + 500]
        dist = 64 - (A @ B.T + (1 - A) @ (1 - B).T)
        ii, jj = np.where(dist <= NEAR_THRESHOLD)
        for a_loc, b in zip(ii, jj):
            a = a_loc + i
            if a < b:
                pairs.append((a, b, int(dist[a_loc, b])))
    pdf = pd.DataFrame(pairs, columns=["a", "b", "dist"])
    pdf["path_a"] = df.path.values[pdf.a]
    pdf["path_b"] = df.path.values[pdf.b]
    pdf["split_a"] = df.split.values[pdf.a]
    pdf["split_b"] = df.split.values[pdf.b]
    pdf["label_a"] = df.label.values[pdf.a]
    pdf["label_b"] = df.label.values[pdf.b]
    OUT.mkdir(exist_ok=True)
    pdf.drop(columns=["a", "b"]).to_csv(OUT / "near_duplicate_pairs.csv", index=False)

    cross = pdf[pdf.split_a != pdf.split_b]
    test_idx = set(pdf.b[(pdf.split_b == "Testing") & (pdf.split_a == "Training")])
    test_idx |= set(pdf.a[(pdf.split_a == "Testing") & (pdf.split_b == "Training")])
    print(f"\n== Near duplicates (pHash distance <= {NEAR_THRESHOLD}) ==")
    print("Total near-duplicate pairs:", len(pdf))
    print("  within Training:", int(((pdf.split_a == "Training") & (pdf.split_b == "Training")).sum()))
    print("  within Testing: ", int(((pdf.split_a == "Testing") & (pdf.split_b == "Testing")).sum()))
    print("  across Train/Test:", len(cross))
    print("Test images with a near-duplicate in Training:", len(test_idx),
          f"({100 * len(test_idx) / (df.split == 'Testing').sum():.1f}% of test set)")
    print("Pairs whose labels DISAGREE:", int((pdf.label_a != pdf.label_b).sum()))
    print("\nPairs saved to reports/near_duplicate_pairs.csv")


if __name__ == "__main__":
    main()