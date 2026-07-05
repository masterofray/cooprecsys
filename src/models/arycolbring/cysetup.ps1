#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

Write-Host "========================================="
Write-Host "BUILDING CYTHON MODULES (Windows)"
Write-Host "========================================="
Write-Host ""

python --version
Write-Host ""

if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}

if (Test-Path "CLproximity") {
    Get-ChildItem -Path "CLproximity" -Filter "*.c"   -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -Path "CLproximity" -Filter "*.pyd" -ErrorAction SilentlyContinue | Remove-Item -Force
}

Write-Host ""
python .\cysetup.py build_ext --inplace
if ($LASTEXITCODE -ne 0) {
    Write-Error "cysetup.py build_ext failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "========================================="
Write-Host "BUILD FINISHED"
Write-Host "========================================="
