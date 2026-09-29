from pydantic import BaseModel


class BatchRequest(BaseModel):
    study_ids: list[str] | None = None


class LabelUpdate(BaseModel):
    """Ручная правка разметки: область, сторона, итоговый класс и критерии.

    violation_type — коды критериев через ';' (например
    'hip_positioning_incorrect;hip_roi_incorrect'), либо 'none' для
    качественного снимка.
    """

    anatomical_region: str
    side: str
    quality_class: int
    violation_type: str
    notes: str = ""
