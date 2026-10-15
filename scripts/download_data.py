"""Download the Brain Tumor MRI Dataset (Masoud Nickparvar) from Kaggle."""
import subprocess
import sys
from pathlib import Path

SLUG = "masoudnickparvar/brain-tumor-mri-dataset"
DEST = Path("data/raw")
EXPECTED = {"glioma", "meningioma", "notumor", "pituitary"}
EXT = {".jpg", ".jpeg", ".png"}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if not (DEST / "Training").exists():
        cmd = ["kaggle", "datasets", "download", "-d", SLUG, "-p", str(DEST), "--unzip"]
        print("Running:", " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            sys.exit("Download failed. Paste the full error to me.")
    else:
        print("data/raw/Training already exists, skipping download.")

    total = 0
    for split in ("Training", "Testing"):
        classes = {p.name for p in (DEST / split).iterdir() if p.is_dir()}
        if classes != EXPECTED:
            sys.exit(f"Unexpected folders in {split}: {classes}")
        for cls in sorted(classes):
            n = sum(1 for p in (DEST / split / cls).iterdir() if p.suffix.lower() in EXT)
            total += n
            print(f"{split}/{cls:12s} {n}")
    print("Total images:", total)


if __name__ == "__main__":
    main()