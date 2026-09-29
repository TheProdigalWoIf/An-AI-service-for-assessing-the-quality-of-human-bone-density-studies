from __future__ import annotations

import csv
import hashlib
import io
import tempfile
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image

from dxa_qc.config import settings
from dxa_qc.dicom import normalize_array, normalized_pixels, preview_data_url, read_dataset, save_preview
from dxa_qc.geometry import build_qc_annotation
from dxa_qc.inference import analyze_file, analyze_uploaded_image, model_available
from dxa_qc.repository import LabelStore, dicom_files, pseudo_label, studies
from dxa_qc.schemas import BatchRequest, LabelUpdate


router = APIRouter(prefix="/api")
label_store = LabelStore(settings.labels_path)
UPLOAD_LIMIT = 25 * 1024 * 1024
MAX_BATCH_FILES = 50
UPLOAD_SUFFIXES = {".dcm", ".dicom", ".jpg", ".jpeg", ".png", ".webp"}


def safe_study(study_id: str) -> Path:
    candidate = (settings.data_root / study_id).resolve()
    if settings.data_root not in candidate.parents or not candidate.is_dir():
        raise HTTPException(404, "Study not found")
    return candidate


def select_studies(study_ids: list[str] | None) -> list[Path]:
    return [safe_study(item) for item in study_ids] if study_ids else studies(settings.data_root)


def batch_results(study_ids: list[str] | None) -> list[dict]:
    return [
        analyze_file(path, settings.data_root).to_dict()
        for directory in select_studies(study_ids)
        for path in dicom_files(directory)
    ]


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "data_available": settings.data_root.exists(), "labels_available": settings.labels_path.exists(), "model_available": model_available()}


@router.get("/studies")
def list_studies() -> dict:
    items = [{"id": path.name, "images": len(dicom_files(path))} for path in studies(settings.data_root)]
    return {"items": items, "total": len(items), "images": sum(item["images"] for item in items)}


@router.get("/studies/{study_id}")
def get_study(study_id: str) -> dict:
    directory = safe_study(study_id)
    labels = {item.image_uid: item for item in label_store.read()}
    items = []
    for path in dicom_files(directory):
        row = analyze_file(path, settings.data_root).to_dict()
        label = labels.get(row["image_uid"])
        if label:
            row.update(
                anatomical_region=label.anatomical_region,
                side=label.side,
                quality_class=label.quality_class if label.quality_class is not None else row["quality_class"],
                violation_type=label.violation_type if label.quality_class is not None else row["violation_type"],
                label_status="unlabeled" if label.quality_class is None else "labeled",
                notes=label.notes,
            )
        else:
            row.update(label_status="missing", notes="")
        row["preview_url"] = f"/api/preview/{study_id}/{row['image_uid']}"
        items.append(row)
    return {"study_id": study_id, "items": items}


@router.get("/preview/{study_id}/{image_uid}")
def preview(study_id: str, image_uid: str):
    directory = safe_study(study_id)
    for path in dicom_files(directory):
        result = analyze_file(path, settings.data_root)
        if result.image_uid == image_uid:
            cache = settings.preview_root / study_id / f"{path.stem}.jpg"
            if not cache.exists():
                save_preview(path, cache)
            return FileResponse(cache, media_type="image/jpeg")
    raise HTTPException(404, "Image not found")


@router.post("/analyze")
def analyze(request: BatchRequest) -> dict:
    items = batch_results(request.study_ids)
    return {"items": items, "processed": len(items), "failures": sum(x["processing_status"] == "Failure" for x in items)}


async def process_uploaded_file(file: UploadFile) -> dict:
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise HTTPException(415, "Поддерживаются DICOM, JPEG, PNG и WebP")
    data = await file.read(UPLOAD_LIMIT + 1)
    await file.close()
    if not data:
        raise HTTPException(422, "Файл пуст")
    if len(data) > UPLOAD_LIMIT:
        raise HTTPException(413, "Размер файла превышает 25 МБ")

    digest = hashlib.sha256(data).hexdigest()[:24]
    preview = None
    annotation = None
    if suffix in {".dcm", ".dicom"}:
        with tempfile.TemporaryDirectory(prefix="dxa-upload-") as directory:
            path = Path(directory) / f"upload{suffix}"
            path.write_bytes(data)
            result = analyze_file(path, path.parent)
            if result.processing_status == "Success":
                preview = preview_data_url(path)
                annotation = build_qc_annotation(normalized_pixels(read_dataset(path)), result.anatomical_region)
        # Never return identifiers embedded in a user-supplied DICOM file.
        result.path_to_study = filename
        result.study_uid = "browser_upload"
        result.image_uid = digest
    else:
        result = analyze_uploaded_image(data, filename)
        if result.processing_status == "Success":
            with Image.open(BytesIO(data)) as image:
                pixels = normalize_array(image.convert("RGB"))
            annotation = build_qc_annotation(pixels, result.anatomical_region)
    if result.processing_status != "Success":
        raise HTTPException(422, "Не удалось прочитать изображение")
    # Угол калибровочной модели (по остистым отросткам, из задания) приоритетнее;
    # геометрический угол оверлея остаётся запасным значением.
    if annotation and result.angle_degrees is None:
        result.angle_degrees = annotation["angle_degrees"]
    return {
        "result": result.to_dict(),
        "preview_data_url": preview,
        "annotation": annotation,
        "notice": "Исследовательская модель; результат не является медицинским заключением.",
    }


@router.post("/check-upload")
async def check_upload(file: UploadFile = File(...)) -> dict:
    return await process_uploaded_file(file)


@router.post("/upload-preview")
async def upload_preview(file: UploadFile = File(...)) -> dict:
    filename = Path(file.filename or "upload.dcm").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".dcm", ".dicom"}:
        raise HTTPException(415, "Предварительное декодирование используется только для DICOM")
    data = await file.read(UPLOAD_LIMIT + 1)
    await file.close()
    if not data:
        raise HTTPException(422, "Файл пуст")
    if len(data) > UPLOAD_LIMIT:
        raise HTTPException(413, "Размер файла превышает 25 МБ")
    try:
        with tempfile.TemporaryDirectory(prefix="dxa-preview-") as directory:
            path = Path(directory) / f"preview{suffix}"
            path.write_bytes(data)
            preview = preview_data_url(path)
    except Exception as exc:
        raise HTTPException(422, "Не удалось построить DICOM-предпросмотр") from exc
    return {"filename": filename, "preview_data_url": preview}


@router.post("/check-uploads")
async def check_uploads(files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(422, "Не выбраны файлы")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(413, f"За один раз можно загрузить не более {MAX_BATCH_FILES} файлов")
    items = []
    for file in files:
        filename = Path(file.filename or "upload").name
        try:
            item = await process_uploaded_file(file)
            item["filename"] = filename
            items.append(item)
        except HTTPException as exc:
            items.append({"filename": filename, "error": str(exc.detail)})
    failures = sum("error" in item for item in items)
    return {"items": items, "processed": len(items) - failures, "failures": failures}


@router.post("/export.csv")
def export_csv(request: BatchRequest):
    columns = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class", "violation_type", "processing_status", "time_of_processing"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(batch_results(request.study_ids))
    return Response(output.getvalue(), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": "attachment; filename=dxa_quality_report.csv"})


@router.get("/annotations")
def annotations() -> dict:
    labels = label_store.read()
    items = [item.to_dict() for item in labels]
    return {
        "items": items,
        "total": len(items),
        "labeled": sum(item.quality_class is not None for item in labels),
        "unlabeled": sum(item.quality_class is None for item in labels),
        "excluded": sum(item.exclude_from_training for item in labels),
    }


@router.post("/annotations/bootstrap")
def bootstrap_annotations() -> dict:
    if settings.labels_path.exists():
        raise HTTPException(409, "Labels already exist")
    labels = [pseudo_label(analyze_file(path, settings.data_root)) for directory in studies(settings.data_root) for path in dicom_files(directory)]
    label_store.write(labels)
    return {"created": len(labels), "unlabeled": len(labels)}


@router.put("/annotations/{image_uid}")
def update_annotation(image_uid: str, request: LabelUpdate) -> dict:
    try:
        item = label_store.update(image_uid, request.anatomical_region, request.side, request.quality_class,
                                  request.violation_type, request.notes)
    except KeyError as exc:
        raise HTTPException(404, "Label not found; bootstrap labels first") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"item": item.to_dict()}
