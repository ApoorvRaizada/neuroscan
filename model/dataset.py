"""Dataset and transforms. Reads the CSV manifests made by scripts/make_splits.py."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool, size: int = 224):
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose([
            T.Resize((size, size)),
            T.RandomRotation(10),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), norm])


class MRIDataset(Dataset):
    def __init__(self, split: str, root: str = ".", train: bool = False, size: int = 224):
        self.root = Path(root)
        self.df = pd.read_csv(self.root / "data" / "splits" / f"{split}.csv")
        self.tf = build_transforms(train, size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row.path).convert("RGB")  # grayscale/RGBA -> 3 channels
        return self.tf(img), CLASS_TO_IDX[row.label]