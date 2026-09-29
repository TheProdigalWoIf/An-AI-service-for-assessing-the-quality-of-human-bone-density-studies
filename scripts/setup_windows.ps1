$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher не найден. Установите Python 3.12 x64 с python.org и включите опцию 'py launcher'."
}

$version = & py -3.12 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
if ($LASTEXITCODE -ne 0 -or $version -ne "3.12") {
    throw "Нужен Python 3.12 x64. Проверка 'py -3.12' не пройдена."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py -3.12 -m venv .venv
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt

if (-not (Test-Path "annotations\labels.csv")) {
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    & $Python scripts\bootstrap_labels.py
}

Write-Host "Установка завершена. Запустите run_windows.bat."
