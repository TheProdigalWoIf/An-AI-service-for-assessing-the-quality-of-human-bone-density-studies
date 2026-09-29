from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


# Анатомическая область и сторона — независимые выходы модели:
# позвоночник всегда side='none', бедро — right/left.
REGIONS = ("lumbar_spine", "proximal_femur")

SIDES = ("none", "right", "left")

# Пять экспертных критериев (схема «снимок -> область -> критерии -> решение»):
# позвоночник — укладка, ось, артефакты; бедро — ротация/позиционирование, ROI.
VIOLATIONS = (
    "spine_positioning_incorrect",
    "spine_axis_misalignment",
    "spine_artifact_or_object",
    "hip_positioning_incorrect",
    "hip_roi_incorrect",
)


def allowed_violations(region: str) -> set[str]:
    """Критерии, применимые к области: у позвоночника три, у бедра два."""
    if region == "lumbar_spine":
        return {
            "spine_positioning_incorrect",
            "spine_axis_misalignment",
            "spine_artifact_or_object",
        }
    if region == "proximal_femur":
        return {
            "hip_positioning_incorrect",
            "hip_roi_incorrect",
        }
    return set()


class DXAMultiTaskModel(nn.Module):
    """ResNet18 + четыре головы: область, сторона, качество, 5 критериев."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.region_head = nn.Linear(features, len(REGIONS))
        self.side_head = nn.Linear(features, len(SIDES))
        self.quality_head = nn.Linear(features, 1)
        self.violation_head = nn.Linear(features, len(VIOLATIONS))

    def forward(self, x):
        features = self.backbone(x)
        return (
            self.region_head(features),
            self.side_head(features),
            self.quality_head(features).squeeze(1),
            self.violation_head(features),
        )


class Letterbox224:
    """Масштабирование без искажения пропорций на чёрный квадрат 224x224."""

    def __call__(self, image):
        width, height = image.size
        scale = min(224 / width, 224 / height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (224, 224), color=0)
        canvas.paste(image, ((224 - new_width) // 2, (224 - new_height) // 2))
        return canvas


def image_transform(training: bool = False):
    operations: list = [
        transforms.ToPILImage(),
        Letterbox224(),
    ]
    if training:
        operations.append(transforms.RandomAutocontrast(p=0.25))
    operations.extend([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transforms.Compose(operations)


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = DXAMultiTaskModel(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def violation_vector(value: str) -> np.ndarray:
    """Строка кодов через ';' -> вектор по VIOLATIONS. 'none'/'' -> нули."""
    if not value or value.strip() == "none":
        selected = set()
    else:
        selected = {
            item.strip()
            for item in value.split(";")
            if item.strip() and item.strip() != "none"
        }
    unknown = selected - set(VIOLATIONS)
    if unknown:
        raise ValueError(f"Unknown violation codes: {sorted(unknown)}")
    return np.asarray([float(name in selected) for name in VIOLATIONS], dtype=np.float32)
