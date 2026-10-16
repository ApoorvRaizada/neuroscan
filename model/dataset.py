"""Dataset. Reads the CSV manifests made by scripts/make_splits.py."""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from model.preprocess import CLASSES, build_transforms  # noqa: F401 (re-exported)

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class MRIDataset(Dataset):
    def __init__(self, split: str, root: str = ".", train: bool = False, size: int = 224):
        self.root = Path(root)
        self.df = pd.read_csv(self.root / "data" / "splits" / f"{split}.csv")
        self.tf = build_transforms(train, size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row.path).convert("RGB")
        return self.tf(img), CLASS_TO_IDX[row.label]