import unittest
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from dxa_qc.main import app


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_studies_contract(self):
        payload = self.client.get("/api/studies").json()
        self.assertIn("items", payload)
        self.assertIn("images", payload)

    def test_missing_study(self):
        self.assertEqual(self.client.get("/api/studies/does-not-exist").status_code, 404)

    def test_upload_png(self):
        stream = BytesIO()
        Image.new("L", (64, 64), color=128).save(stream, format="PNG")
        response = self.client.post(
            "/api/check-upload",
            files={"file": ("sample.png", stream.getvalue(), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()["result"]
        self.assertEqual(payload["processing_status"], "Success")
        # Основной путь — мультитаск-модель (resnet18_multitask); без чекпойнта
        # API откатывается на резервные правила qc.py.
        self.assertIn(payload["model_source"], {"resnet18_multitask", "rules"})
        required = {
            "path_to_study", "study_uid", "image_uid", "anatomical_region",
            "side", "quality_class", "violation_type", "processing_status",
            "time_of_processing", "decision_status",
        }
        self.assertTrue(required.issubset(payload))

    def test_upload_rejects_unknown_format(self):
        response = self.client.post(
            "/api/check-upload",
            files={"file": ("sample.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)

    def test_upload_dicom_returns_pixel_preview(self):
        # Детерминированный выбор: первый по порядку снимок ПОЗВОНОЧНИКА из
        # тестового набора сайта (калибровочная модель определяет часть тела,
        # для позвоночника ожидаем разметку с контрольными уровнями).
        path = next(p for p in sorted(Path("Исследования_тест").rglob("*.dcm")) if p.name == "п.dcm")
        response = self.client.post(
            "/api/check-upload",
            files={"file": (path.name, path.read_bytes(), "application/dicom")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["preview_data_url"].startswith("data:image/jpeg;base64,"))
        annotation = payload["annotation"]
        if payload["result"]["anatomical_region"] == "lumbar_spine":
            self.assertEqual(annotation["kind"], "lumbar_geometry")
            self.assertEqual(len(annotation["guides"]), 4)
        else:  # модель отнесла снимок к бедру — геометрия бедра без уровней
            self.assertEqual(annotation["kind"], "hip_geometry")

        preview_response = self.client.post(
            "/api/upload-preview",
            files={"file": (path.name, path.read_bytes(), "application/dicom")},
        )
        self.assertEqual(preview_response.status_code, 200)
        self.assertTrue(preview_response.json()["preview_data_url"].startswith("data:image/jpeg;base64,"))

    def test_batch_upload_processes_multiple_files_independently(self):
        first, second = BytesIO(), BytesIO()
        Image.new("L", (64, 64), color=90).save(first, format="PNG")
        Image.new("L", (64, 64), color=180).save(second, format="PNG")
        response = self.client.post(
            "/api/check-uploads",
            files=[
                ("files", ("first.png", first.getvalue(), "image/png")),
                ("files", ("second.png", second.getvalue(), "image/png")),
                ("files", ("broken.txt", b"bad", "text/plain")),
            ],
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["processed"], 2)
        self.assertEqual(payload["failures"], 1)
        self.assertEqual(len(payload["items"]), 3)
