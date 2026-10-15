"""Load one batch from the train split and run both networks (CPU)."""
import torch
from torch.utils.data import DataLoader

from model.dataset import MRIDataset
from model.networks import BaselineCNN, build_resnet18

ds = MRIDataset("train", train=True)
x, y = next(iter(DataLoader(ds, batch_size=8, shuffle=True)))
print("train images:", len(ds), "| batch:", tuple(x.shape), "| labels:", y.tolist())

for name, net in [("BaselineCNN", BaselineCNN()), ("ResNet18", build_resnet18(pretrained=True))]:
    params = sum(p.numel() for p in net.parameters())
    out = net(x)
    print(f"{name}: params={params:,} output={tuple(out.shape)}")