"""Grad-CAM for ResNet18 (target layer: last residual block)."""
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        self.acts = out.detach()
        out.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x, class_idx=None):
        """x: (1,3,H,W). Returns (heatmap HxW in [0,1], class index, probabilities)."""
        with torch.enable_grad():
            self.model.zero_grad()
            logits = self.model(x)
            probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
            if class_idx is None:
                class_idx = int(logits.argmax(1))
            logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx, probs