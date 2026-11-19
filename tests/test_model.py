import pytest
import torch

from dynow_vision.model import build_model, load_checkpoint, save_checkpoint
from dynow_vision.train import load_config

CFG = load_config("configs/phase1.yaml")


@pytest.fixture(scope="module")
def model():
    return build_model(CFG)


def test_forward_pass_and_freezing(model):
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 2)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    model.train()
    assert not model.backbone.training  # frozen backbone stays in eval mode


def test_checkpoint_roundtrip(model, tmp_path):
    model.eval()
    path = tmp_path / "c.pt"
    save_checkpoint(path, model, CFG, {"epoch": 0, "history": [], "best": {}, "run_name": "t"})
    x = torch.randn(2, 3, 224, 224)
    restored, ck = load_checkpoint(path, "cpu", backbone=model.backbone)
    assert torch.allclose(model(x), restored(x), atol=1e-6)
    assert ck["classes"] == ["clear", "snow"] and len(ck["checkpoint_sha256"]) == 64
    assert path.stat().st_size < 5_000_000  # head only, backbone not duplicated


def test_missing_checkpoint():
    with pytest.raises(FileNotFoundError):
        load_checkpoint("nope.pt")
