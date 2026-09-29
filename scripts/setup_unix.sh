#!/bin/sh
# Подготовка окружения на macOS/Linux: создаёт .venv и ставит зависимости.
set -eu
cd "$(dirname "$0")/.."

if command -v python3.12 >/dev/null 2>&1; then
    PY=python3.12
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "Python не найден. Установите Python 3.12+ с python.org или: brew install python@3.12"
    exit 1
fi

if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
    echo "Нужен Python 3.12+, найден: $($PY --version). Установите: brew install python@3.12"
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    "$PY" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [ ! -f "annotations/labels.csv" ]; then
    PYTHONPATH=src .venv/bin/python scripts/bootstrap_labels.py
fi

echo "Установка завершена. Запустите run_mac.command или: sh scripts/run.sh"
