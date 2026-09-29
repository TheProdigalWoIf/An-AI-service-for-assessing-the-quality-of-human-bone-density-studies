from __future__ import annotations

import warnings
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


# The supplied anonymized files contain overlong, but stable and readable UIDs.
warnings.filterwarnings("ignore", message="Invalid value for VR UI.*", module="pydicom.*")


def read_dataset(path: Path, pixels: bool = True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydicom.dcmread(path, stop_before_pixels=not pixels, force=True)


def normalize_array(source: np.ndarray) -> np.ndarray:
    image = np.asarray(source, dtype=np.float32)
    if image.ndim == 3:
        image = np.mean(image[..., :3], axis=2)
    if image.ndim != 2 or min(image.shape) < 32:
        raise ValueError("Expected a two-dimensional image of at least 32x32 pixels")
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def normalized_pixels(dataset) -> np.ndarray:
    image = normalize_array(dataset.pixel_array)
    if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
        image = 1.0 - image
    return image


def save_preview(path: Path, destination: Path) -> None:
    dataset = read_dataset(path)
    image = Image.fromarray((normalized_pixels(dataset) * 255).astype(np.uint8), mode="L")
    image.thumbnail((960, 960), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, "JPEG", quality=90, optimize=True)


def preview_data_url(path: Path) -> str:
    dataset = read_dataset(path)
    image = Image.fromarray((normalized_pixels(dataset) * 255).astype(np.uint8), mode="L")
    image.thumbnail((960, 960), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
