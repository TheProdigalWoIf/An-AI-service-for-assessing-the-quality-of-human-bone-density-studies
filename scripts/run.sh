#!/bin/sh
# Запуск DXA QC Lab на macOS/Linux через локальное окружение .venv (без Docker).
set -eu
cd "$(dirname "$0")/.."

if [ ! -x ".venv/bin/python" ]; then
    echo "Окружение не создано. Сначала запустите: sh scripts/setup_unix.sh"
    exit 1
fi
if [ ! -f "models/dxa_qc_resnet18.pt" ]; then
    echo "Не найден файл модели models/dxa_qc_resnet18.pt"
    exit 1
fi

export PYTHONPATH="$PWD/src"
echo "Интерфейс: http://localhost:1639 (остановка: Ctrl+C)"
exec .venv/bin/python -m uvicorn dxa_qc.main:app --host 127.0.0.1 --port 1639
