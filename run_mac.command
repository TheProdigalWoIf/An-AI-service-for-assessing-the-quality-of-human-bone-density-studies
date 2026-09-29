#!/bin/sh
# DXA QC Lab — запуск на macOS двойным щелчком (аналог run_windows.bat).
# При первом запуске сам создаёт окружение и ставит зависимости (5–10 минут,
# качается ~250 МБ), дальше стартует быстро.
set -eu
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    sh scripts/setup_unix.sh
fi

exec sh scripts/run.sh
