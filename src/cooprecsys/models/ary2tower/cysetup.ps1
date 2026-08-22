#!/usr/bin/env pwsh
$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir '..\..\..')).Path
if (Test-Path (Join-Path $repoRoot 'src\cooprecsys')) { $sourceRoot = Join-Path $repoRoot 'src' }
elseif (Test-Path (Join-Path $repoRoot 'cooprecsys')) { $sourceRoot = $repoRoot }
else { throw 'ERROR: cooprecsys source tree not found' }
Set-Location $repoRoot
$env:PYTHONPATH = "$sourceRoot;$env:PYTHONPATH"
& python (Join-Path $scriptDir 'a2tcysetup.py') build_ext --inplace
if ($LASTEXITCODE -ne 0) { throw "a2tcysetup.py build_ext failed: $LASTEXITCODE" }
& python -c "from cooprecsys.models.ary2tower.towers import backend_info; print('[OK] backend ->', backend_info()); if (-not (backend_info()['compiled'])) { exit 1 }"
