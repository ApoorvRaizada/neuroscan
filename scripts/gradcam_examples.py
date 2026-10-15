"""Save Grad-CAM overlays for correct and incorrect ResNet18 test predictions."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from model.dataset import CLASSES, MRIDataset
from model.gradcam import GradCAM
from model.networks import build_resnet18

SEED = 42
N = 6
OUT = Path("reports/figures")


def draw(items, ds, cam, device, title, path):
    fig, axes = plt.subplots(2, len(items), figsize=(2.6 * len(items), 5.6), squeeze=False)
    for k, (i, true, pred) in enumerate(items):
        x, _ = ds[i]
        heat, _, probs = cam(x.unsqueeze(0).to(device), class_idx=pred)
        img = Image.open(ds.root / ds.df.iloc[i].path).convert("RGB").resize((224, 224))
        axes[0, k].imshow(img)
        axes[1, k].imshow(img)
        axes[1, k].imshow(heat, cmap="jet", alpha=0.4)
        axes[0, k].set_title(f"true: {CLASSES[true]}\npred: {CLASSES[pred]} ({probs[pred]:.2f})", fontsize=8)
        axes[0, k].axis("off")
        axes[1, k].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Saved", path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_resnet18(pretrained=False)
    net.load_state_dict(torch.load("runs/resnet18/best.pt", map_location=device))
    net.to(device).eval()
    cam = GradCAM(net, net.layer4[-1])

    ds = MRIDataset("test", train=False)
    preds = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=64, shuffle=False, num_workers=2):
            preds += net(x.to(device)).argmax(1).tolist()
    preds = np.array(preds)
    true = np.array([CLASSES.index(l) for l in ds.df.label])

    rng = np.random.default_rng(SEED)
    right = rng.choice(np.where(preds == true)[0], N, replace=False)
    wrong = np.where(preds != true)[0]
    # show tumors predicted as "no tumor" first: the costly mistake
    wrong = sorted(wrong, key=lambda i: preds[i] != CLASSES.index("notumor"))[:N]

    OUT.mkdir(parents=True, exist_ok=True)
    draw([(i, true[i], preds[i]) for i in right], ds, cam, device,
         "Correct predictions (random sample)", OUT / "gradcam_correct.png")
    draw([(i, true[i], preds[i]) for i in wrong], ds, cam, device,
         "Errors (missed tumors first)", OUT / "gradcam_errors.png")


if __name__ == "__main__":
    main()