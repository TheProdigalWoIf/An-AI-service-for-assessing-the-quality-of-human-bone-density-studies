FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src DXA_DATA_ROOT=/data DXA_LABELS_PATH=/annotations/labels.csv
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
COPY models ./models
EXPOSE 8000
CMD ["uvicorn", "dxa_qc.main:app", "--host", "0.0.0.0", "--port", "8000"]
