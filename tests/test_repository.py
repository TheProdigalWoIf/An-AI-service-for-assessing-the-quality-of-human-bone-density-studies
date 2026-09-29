import tempfile
import unittest
from pathlib import Path

from dxa_qc.models import QCResult
from dxa_qc.repository import (
    CRITERION_BY_VIOLATION,
    LABEL_COLUMNS,
    LabelStore,
    validate_label,
    pseudo_label,
)


def make_label(**overrides):
    """Размеченная строка бедра с нарушенной ротацией."""
    base = dict(
        path_to_study="3\\3_2\\пб.dcm",
        study_uid="3",
        image_uid="3",
        anatomical_region="proximal_femur",
        side="right",
        quality_class=1,
        violation_type="hip_positioning_incorrect",
        notes="",
        exclude_from_training=False,
    )
    base.update(overrides)
    return type("Label", (), base)  # noqa: совместимый по атрибутам объект


class LabelStoreTests(unittest.TestCase):
    def test_roundtrip_expert_schema(self):
        """Запись и чтение схемы из пяти колонок критериев."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            store = LabelStore(path)
            store.write([
                make_label(),
                make_label(path_to_study="3\\3_1\\п.dcm", image_uid="2",
                           anatomical_region="lumbar_spine", side="none",
                           quality_class=0, violation_type="none"),
            ])

            text = path.read_text(encoding="utf-8-sig")
            self.assertEqual(text.splitlines()[0].split(";"), LABEL_COLUMNS)

            rows = store.read()
            self.assertEqual(rows[0].violation_type, "hip_positioning_incorrect")
            self.assertEqual(rows[1].violation_type, "none")

            # Колонка критерия восстанавливается из кода нарушения.
            lines = text.splitlines()
            self.assertEqual(lines[1].split(";")[LABEL_COLUMNS.index("hip_positioning_rotation")], "1")

    def test_region_alias_hip(self):
        """Экспертское 'hip' канонизируется в proximal_femur."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            path.write_text(
                "path_to_study;study_uid;image_uid;anatomical_region;side;quality_class;"
                "correct_positioning;spine_axis_alignment;foreign_objects_artifacts;"
                "hip_positioning_rotation;hip_roi_correctness;notes;exclude_from_training\n"
                "1\\1_1\\x.dcm;1;1;hip;left;1;;;;1;0;;False\n",
                encoding="utf-8-sig",
            )
            rows = LabelStore(path).read()
            self.assertEqual(rows[0].anatomical_region, "proximal_femur")
            self.assertEqual(rows[0].violation_type, "hip_positioning_incorrect")

    def test_unlabeled_row_kept_and_skipped_by_criteria(self):
        """Неразмеченная строка (пустой quality_class) читается как None."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            path.write_text(
                "path_to_study;study_uid;image_uid;anatomical_region;side;quality_class;"
                "correct_positioning;spine_axis_alignment;foreign_objects_artifacts;"
                "hip_positioning_rotation;hip_roi_correctness;notes;exclude_from_training\n"
                "1\\1_2\\x.dcm;1;2;hip;left;;;;;;;эндопротез;False\n",
                encoding="utf-8-sig",
            )
            rows = LabelStore(path).read()
            self.assertIsNone(rows[0].quality_class)
            self.assertEqual(rows[0].violation_type, "none")

    def test_comma_delimiter_still_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            path.write_text(
                "path_to_study,study_uid,image_uid,anatomical_region,side,quality_class,"
                "correct_positioning,spine_axis_alignment,foreign_objects_artifacts,"
                "hip_positioning_rotation,hip_roi_correctness,notes,exclude_from_training\n"
                "1\\1_1\\x.dcm,1,1,lumbar_spine,none,1,1,0,0,,,note,False\n",
                encoding="utf-8-sig",
            )
            rows = LabelStore(path).read()
            self.assertEqual(rows[0].violation_type, "spine_positioning_incorrect")

    def test_validate_rejects_inconsistent_and_foreign(self):
        with self.assertRaises(ValueError):  # хороший снимок с критерием
            validate_label(make_label(quality_class=0))
        with self.assertRaises(ValueError):  # плохой без критериев
            validate_label(make_label(quality_class=1, violation_type="none"))
        with self.assertRaises(ValueError):  # критерий позвоночника у бедра
            validate_label(make_label(violation_type="spine_axis_misalignment"))
        with self.assertRaises(ValueError):  # неизвестный код
            validate_label(make_label(violation_type="low_contrast"))
        with self.assertRaises(ValueError):  # бедро без стороны
            validate_label(make_label(side="none"))
        with self.assertRaises(ValueError):  # позвоночник со стороной
            validate_label(make_label(anatomical_region="lumbar_spine", side="right",
                                      violation_type="spine_axis_misalignment"))

    def test_update_annotation(self):
        result = QCResult(path_to_study="p", study_uid="s", image_uid="i",
                          anatomical_region="lumbar_spine", quality_class=0,
                          violation_type="none", processing_status="Success",
                          time_of_processing=0.1)
        with tempfile.TemporaryDirectory() as directory:
            store = LabelStore(Path(directory) / "labels.csv")
            store.write([pseudo_label(result)])
            self.assertIsNone(store.read()[0].quality_class)  # бутстрэп = неразмечено
            updated = store.update("i", "lumbar_spine", "none", 1, "spine_axis_misalignment", "проверено")
            self.assertEqual(updated.quality_class, 1)
            self.assertEqual(updated.violation_type, "spine_axis_misalignment")
            self.assertEqual(store.read()[0].notes, "проверено")

    def test_criterion_violation_mapping_is_complete(self):
        self.assertEqual(len(CRITERION_BY_VIOLATION), 5)
        self.assertEqual(
            set(CRITERION_BY_VIOLATION.values()),
            {"correct_positioning", "spine_axis_alignment", "foreign_objects_artifacts",
             "hip_positioning_rotation", "hip_roi_correctness"},
        )


if __name__ == "__main__":
    unittest.main()
