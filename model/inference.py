"""Load the model once, run predictions."""
import torch
from PIL import Image

from model.networks import build_resnet18
from model.preprocess import CLASSES, build_transforms


class Predictor:
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.net = build_resnet18(pretrained=False)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state)
        self.net.to(self.device).eval()
        self.tf = build_transforms(train=False)

    def predict(self, img: Image.Image) -> dict:
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.net(x), 1)[0].tolist()
        top = max(range(len(probs)), key=probs.__getitem__)
        return {
            "label": CLASSES[top],
            "confidence": probs[top],
            "probabilities": dict(zip(CLASSES, probs)),
        }