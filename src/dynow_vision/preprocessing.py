"""Image validation, resizing, normalization and augmentation."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # statistics DINOv2 was trained with
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 14
MIN_SIDE = 64
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_image(path: str | Path) -> tuple[bool, str]:
    """Return (ok, reason). Fully decodes the file so truncated images are caught."""
    path = Path(path)
    if not path.is_file():
        return False, "file does not exist"
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False, f"unsupported extension {path.suffix!r}"
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            w, h = im.size
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        return False, f"cannot decode image: {e}"
    if min(w, h) < MIN_SIDE:
        return False, f"image too small ({w}x{h}, min side {MIN_SIDE})"
    return True, "ok"


def load_rgb(path: str | Path) -> Image.Image:
    """Load an image as RGB, honoring EXIF orientation."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def check_input_size(size: int) -> int:
    if size % PATCH_SIZE != 0:
        raise ValueError(f"input_size must be a multiple of {PATCH_SIZE}, got {size}")
    return size


def eval_transform(size: int = 224) -> T.Compose:
    """Deterministic: resize the whole frame to size x size (no crop, so no sidewalk is cut off)."""
    check_input_size(size)
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform(size: int = 224) -> T.Compose:
    """Mild augmentation. No vertical flip (sidewalks have a fixed orientation) and
    no hue jitter (colour shifts could change the snow/clear appearance)."""
    check_input_size(size)
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.33),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
