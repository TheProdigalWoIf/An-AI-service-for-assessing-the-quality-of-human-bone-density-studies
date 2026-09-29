param(
    [ValidateRange(1, 1000)]
    [int]$Epochs = 8
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Окружение не создано. Сначала запустите setup_windows.bat."
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& $Python scripts\train_model_2.py --epochs $Epochs
& $Python scripts\evaluate_model.py
& $Python scripts\batch_inference.py
