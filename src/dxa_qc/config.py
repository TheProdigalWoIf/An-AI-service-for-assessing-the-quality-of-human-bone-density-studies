from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)  # frozen: конфигурацию нельзя случайно подменить после создания
class Settings:
    package_dir: Path = Path(__file__).resolve().parent
    # Корень данных сайта — ТЕСТОВЫЙ набор калибровки: 20 исследований (44
    # снимка), на которых модель НЕ обучалась; обучающие 80 исследований
    # лежат отдельно в «Исследования_обучение» и интерфейсом не показываются.
    data_root: Path = Path(os.getenv("DXA_DATA_ROOT", Path.cwd() / "Исследования_тест")).resolve()
    labels_path: Path = Path(
        os.getenv("DXA_LABELS_PATH", Path.cwd() / "annotations" / "labels.csv")
    ).resolve()
    preview_root: Path = Path(
        os.getenv("DXA_PREVIEW_ROOT", Path(tempfile.gettempdir()) / "dxa-qc-previews")
    ).resolve()
    # Основная модель контроля качества (scripts/train_model_2.py): область,
    # сторона, класс качества и 5 экспертных критериев. Если чекпойнта нет,
    # inference.py откатывается на резервные правила qc.py.
    model_path: Path = Path(
        os.getenv("DXA_MODEL_PATH", Path.cwd() / "models" / "dxa_qc_resnet18.pt")
    ).resolve()
    # Калибровочная модель (scripts/train_calibration.py) — отдельный артефакт,
    # используется только модулем dxa_qc.calibration.
    calibration_path: Path = Path(
        os.getenv("DXA_CALIBRATION_PATH", Path.cwd() / "models" / "calibration_resnet18.pt")
    ).resolve()

    @property
    def static_dir(self) -> Path:
        return self.package_dir / "static"


settings = Settings()
