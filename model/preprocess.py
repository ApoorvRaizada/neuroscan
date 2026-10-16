"""Shared preprocessing. Used by training, evaluation and the API."""
from torchvision import transforms as T

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
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