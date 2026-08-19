#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"


$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================="
Write-Host "BUILDING CYTHON MODULES (Windows)"
Write-Host "========================================="
Write-Host ""

python --version
Write-Host ""

Write-Host "--- Diagnostics ---"
Write-Host "CWD: $(Get-Location)"
python -c "import sys; print('Python exe:', sys.executable)"
python -c "import setuptools; print('setuptools:', setuptools.__version__)"
python -c "import Cython; print('Cython:', Cython.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
$clPath = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($clPath) {
    Write-Host "cl.exe found: $($clPath.Source)"
} else {
    Write-Host "::warning::cl.exe not found on PATH. If build_ext fails with 'Microsoft Visual C++ ... is required' or 'cl.exe failed', add the 'ilammy/msvc-dev-cmd@v1' action step before this one in the workflow."
}
Write-Host ""

if (-not (Test-Path "cysetup.py")) {
    Write-Error "cysetup.py not found in $(Get-Location). Check the 'working-directory' setting in the workflow step."
    exit 1
}

if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}

# Recursively clean previously-compiled artifacts anywhere under this
# directory (not just the top-level CLtowers folder), so stale
# .pyd/.c files from a prior run can't mask a build failure.
Get-ChildItem -Recurse -Filter "*.c"   -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch '\\build\\' } | Remove-Item -Force
Get-ChildItem -Recurse -Filter "*.pyd" -ErrorAction SilentlyContinue | Remove-Item -Force

Write-Host ""
Write-Host "Running: python .\cysetup.py build_ext --inplace"
python .\cysetup.py build_ext --inplace
if ($LASTEXITCODE -ne 0) {
    Write-Error "cysetup.py build_ext failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "--- Build output (.pyd files produced) ---"
Get-ChildItem -Recurse -Filter "*.pyd" | ForEach-Object { Write-Host $_.FullName }

Write-Host "========================================="
Write-Host "BUILD FINISHED"
Write-Host "========================================="
