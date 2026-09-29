$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Окружение не создано. Сначала запустите setup_windows.bat."
}
if (-not (Test-Path "models\dxa_qc_resnet18.pt")) {
    throw "Не найден файл модели models\dxa_qc_resnet18.pt."
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
Write-Host "Интерфейс: http://localhost:1639"
& $Python -m uvicorn dxa_qc.main:app --host 127.0.0.1 --port 1639
