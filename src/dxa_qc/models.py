from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class QCResult:
    """Результат анализа одного снимка.

    anatomical_region и side разделены: область — lumbar_spine/proximal_femur,
    сторона — none (позвоночник) / right / left (бедро).
    quality_class: 1 = есть нарушение качества, 0 = снимок хороший;
    violation_type — коды нарушенных критериев через ';' либо 'none'.
    decision_status: 'final' — итоговое решение, 'needs_review' —
    требует ручной проверки (плохой снимок без конкретного критерия,
    неопределённая сторона у правил, ошибка чтения).
    """

    path_to_study: str
    study_uid: str
    image_uid: str

    anatomical_region: str
    side: str = "none"

    quality_class: int = 1
    violation_type: str = "none"

    processing_status: str = "Success"
    time_of_processing: float = 0.0
    quality_score: float = 0.0
    angle_degrees: float | None = None
    model_source: str = "rules_v1"
    # Поля калибровочной модели (значения по умолчанию не ломают старые
    # конструкторы и CSV-экспорт с extrasaction="ignore"):
    clinical_finding: str = ""                      # клиническое отклонение — НЕ нарушение качества
    clinical_probability: float | None = None       # P(клиническое отклонение) по отдельной голове
    axis_deviation_deg: float | None = None         # отклонение оси |угол-90°| по остистым отросткам

    decision_status: str = "needs_review"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LabelRecord:
    """Строка annotations/labels.csv — экспертная разметка.

    Схема повторает CSV (разделитель ';'):
    path_to_study;study_uid;image_uid;anatomical_region;side;quality_class;
    correct_positioning;spine_axis_alignment;foreign_objects_artifacts;
    hip_positioning_rotation;hip_roi_correctness;notes;exclude_from_training

    anatomical_region канонизируется при чтении: 'hip' -> 'proximal_femur'.
    Пять колонок критериев сворачиваются в violation_type (коды через ';'),
    quality_class может быть None — строка есть, но ещё не размечена.
    exclude_from_training=True исключает изображение из обучения,
    не удаляя его из данных.
    """

    path_to_study: str
    study_uid: str
    image_uid: str

    anatomical_region: str
    side: str
    quality_class: int | None
    violation_type: str

    notes: str = ""
    exclude_from_training: bool = False

    def to_dict(self) -> dict:
        return asdict(self)
