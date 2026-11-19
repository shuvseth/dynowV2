"""DINOv2 frozen backbone + trainable classification head."""
from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import torch
from torch import nn

from .preprocessing import check_input_size

HUB_REPO = "facebookresearch/dinov2"
EMBED_DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768}


def select_device(pref: str = "auto") -> torch.device:
    """MPS if available (and requested/auto), otherwise CPU."""
    if pref in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "mps":
        print("warning: MPS requested but unavailable; falling back to CPU")
    return torch.device("cpu")


def load_backbone(name: str = "dinov2_vits14") -> nn.Module:
    """Pretrained DINOv2 (weights from Meta, downloaded once to ~/.cache/torch/hub)."""
    if name not in EMBED_DIMS:
        raise ValueError(f"unsupported backbone {name!r}; choose from {list(EMBED_DIMS)}")
    return torch.hub.load(HUB_REPO, name, verbose=False)


class WinterClassifier(nn.Module):
    """Frozen backbone -> CLS embedding -> trainable head -> class logits."""

    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int,
                 hidden: int = 0, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        layers: list[nn.Module] = [nn.Dropout(dropout)]
        if hidden > 0:
            layers += [nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes)]
        else:
            layers += [nn.Linear(embed_dim, num_classes)]
        self.head = nn.Sequential(*layers)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # frozen backbone stays in eval mode
        return self

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def build_model(cfg: dict, backbone: nn.Module | None = None) -> WinterClassifier:
    check_input_size(cfg["input_size"])
    backbone = backbone if backbone is not None else load_backbone(cfg["backbone"])
    return WinterClassifier(backbone, EMBED_DIMS.get(cfg["backbone"], getattr(backbone, "embed_dim", 0)),
                            len(cfg["classes"]), cfg.get("head_hidden", 0), cfg.get("head_dropout", 0.2))


def save_checkpoint(path: str | Path, model: WinterClassifier, cfg: dict, extra: dict,
                    optimizer: torch.optim.Optimizer | None = None) -> None:
    """Only the head is stored: the backbone is the unmodified pretrained DINOv2."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "head_state": model.head.state_dict(),
        "config": cfg,
        "classes": list(cfg["classes"]),
        "backbone": cfg["backbone"],
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        **extra,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu",
                    backbone: nn.Module | None = None) -> tuple[WinterClassifier, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(payload["config"], backbone=backbone)
    model.head.load_state_dict(payload["head_state"])
    payload["checkpoint_sha256"] = file_sha256(path)
    return model.to(device).eval(), payload


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
