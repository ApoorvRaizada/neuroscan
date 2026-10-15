"""Train BaselineCNN or ResNet18 on the leakage-aware splits. Logs to runs/<name>/."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from model.dataset import MRIDataset
from model.networks import BaselineCNN, build_resnet18


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    total, preds, ys = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(y)
            preds += out.argmax(1).tolist()
            ys += y.tolist()
    acc = float(np.mean(np.array(preds) == np.array(ys)))
    return total / len(ys), acc, float(f1_score(ys, preds, average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "resnet18"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use N random train/val images (quick tests)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.name or args.model
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)

    train_ds = MRIDataset("train", train=True)
    val_ds = MRIDataset("val", train=False)
    if args.limit:
        rng = np.random.default_rng(args.seed)
        train_ds = Subset(train_ds, rng.permutation(len(train_ds))[: args.limit].tolist())
        val_ds = Subset(val_ds, rng.permutation(len(val_ds))[: args.limit].tolist())

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers)

    net = BaselineCNN() if args.model == "baseline" else build_resnet18(pretrained=True)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    history, best_f1, best_epoch = [], -1.0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tl, ta, tf = run_epoch(net, train_dl, device, loss_fn, opt)
        vl, va, vf = run_epoch(net, val_dl, device, loss_fn)
        history.append({"epoch": ep, "train_loss": tl, "train_acc": ta, "train_macro_f1": tf,
                        "val_loss": vl, "val_acc": va, "val_macro_f1": vf})
        print(f"ep {ep:2d} | train loss {tl:.3f} f1 {tf:.3f} | val loss {vl:.3f} f1 {vf:.3f}")
        if vf > best_f1:
            best_f1, best_epoch = vf, ep
            torch.save(net.state_dict(), out / "best.pt")

    log = {"args": vars(args), "device": str(device), "torch": torch.__version__,
           "train_seconds": round(time.time() - t0, 1), "best_epoch": best_epoch,
           "best_val_macro_f1": best_f1, "history": history}
    (out / "metrics.json").write_text(json.dumps(log, indent=2))
    print(f"Best val macro-F1 {best_f1:.4f} at epoch {best_epoch}. Saved to {out}")


if __name__ == "__main__":
    main()