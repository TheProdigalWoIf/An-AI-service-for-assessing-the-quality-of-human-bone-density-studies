import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# train_model_2 лежит в scripts/ (не пакет) — подключаем по пути файла.
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_model_2.py"
spec = importlib.util.spec_from_file_location("train_model_2", SCRIPT)
train_model_2 = importlib.util.module_from_spec(spec)
sys.modules["train_model_2"] = train_model_2
spec.loader.exec_module(train_model_2)


def make_rows():
    """80 исследований x 2-3 изображения, ~25% плохих (плохой = есть критерий)."""
    rows = []
    uid = 0
    for study in range(1, 41):
        for image in range(1, 3):
            uid += 1
            bad = uid % 4 == 0
            rows.append(SimpleNamespace(
                study_uid=str(study), image_uid=str(uid),
                anatomical_region="lumbar_spine" if image == 1 else "proximal_femur",
                side="none" if image == 1 else "right",
                quality_class=1 if bad else 0,
                violation_type=(
                    "spine_axis_misalignment" if bad and image == 1
                    else "hip_positioning_incorrect" if bad
                    else "none"
                ),
                path_to_study=f"{study}\\{study}_{image}\\п.dcm",
            ))
    return rows


class GroupSplitTests(unittest.TestCase):
    def test_no_leakage_and_proportions(self):
        rows = make_rows()
        splits = train_model_2.group_split(rows, seed=42)

        self.assertEqual(
            sorted(splits), ["train", "validation"],
            "внутренний test не выделяется: тест — внешняя папка",
        )
        self.assertEqual(
            sum(len(values) for values in splits.values()), len(rows),
            "split must cover every row exactly once",
        )

        studies = {
            name: {row.study_uid for row in values}
            for name, values in splits.items()
        }
        self.assertFalse(studies["train"] & studies["validation"])

        total = len(rows)
        self.assertGreater(len(splits["train"]), 0.7 * total)
        self.assertLess(len(splits["train"]), 0.9 * total)
        self.assertGreater(len(splits["validation"]), 0.1 * total)
        self.assertLess(len(splits["validation"]), 0.3 * total)

        for name, values in splits.items():
            self.assertEqual(
                {row.quality_class for row in values}, {0, 1},
                f"split {name} must contain both quality classes",
            )

    def test_split_is_reproducible(self):
        rows = make_rows()
        first = train_model_2.group_split(rows, seed=7)
        second = train_model_2.group_split(rows, seed=7)
        self.assertEqual(
            [row.image_uid for row in first["validation"]],
            [row.image_uid for row in second["validation"]],
        )


class DerivedQualityTests(unittest.TestCase):
    def test_quality_derived_from_criteria_not_column(self):
        """Итоговое качество считается по критериям и игнорирует quality_class."""
        spine_bad = SimpleNamespace(
            anatomical_region="lumbar_spine",
            violation_type="spine_axis_misalignment;spine_artifact_or_object",
            quality_class=1,
        )
        self.assertEqual(train_model_2.derived_quality(spine_bad), 1)

        femur_bad = SimpleNamespace(
            anatomical_region="proximal_femur", violation_type="hip_roi_incorrect", quality_class=1,
        )
        self.assertEqual(train_model_2.derived_quality(femur_bad), 1)

        # Колонка говорит «плохой», но критериев нет — модель учит «хороший».
        spine_column_only = SimpleNamespace(
            anatomical_region="lumbar_spine", violation_type="none", quality_class=1,
        )
        self.assertEqual(train_model_2.derived_quality(spine_column_only), 0)

        # Критерий чужой области не делает снимок плохим.
        femur_foreign = SimpleNamespace(
            anatomical_region="proximal_femur", violation_type="spine_axis_misalignment", quality_class=1,
        )
        self.assertEqual(train_model_2.derived_quality(femur_foreign), 0)


class ResolvePathTests(unittest.TestCase):
    def test_simplified_path_resolves_by_filename(self):
        """CSV-путь '1\\1_1\\п.dcm' находит реальный файл в глубокой папке."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "1" / "series_002_2_CR" / "A2504381177 DXA" / "CR DXA" / "п.dcm"
            real.parent.mkdir(parents=True)
            real.write_bytes(b"x")

            row = SimpleNamespace(path_to_study="1\\1_1\\п.dcm", study_uid="1")
            self.assertEqual(train_model_2.resolve_image_path(root, row), real)

            row_direct = SimpleNamespace(path_to_study=str(real.relative_to(root)), study_uid="1")
            self.assertEqual(train_model_2.resolve_image_path(root, row_direct), real)

    def test_missing_path_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            row = SimpleNamespace(path_to_study="9\\9_1\\п.dcm", study_uid="9")
            with self.assertRaises(FileNotFoundError):
                train_model_2.resolve_image_path(Path(directory), row)


if __name__ == "__main__":
    unittest.main()
