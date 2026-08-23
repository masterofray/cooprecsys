#!/usr/bin/env pwsh
$ErrorActionPreference = 'Stop'

# 1. Penentuan direktori skrip dan root repositori
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = (Resolve-Path (Join-Path $scriptDir '..\..\..')).Path

# 2. Deteksi lokasi struktur kode sumber (source tree)
if (Test-Path (Join-Path $repoRoot 'src\cooprecsys')) {
    $sourceRoot = Join-Path $repoRoot 'src'
}
elseif (Test-Path (Join-Path $repoRoot 'cooprecsys')) {
    $sourceRoot = $repoRoot
}
else {
    throw 'ERROR: cooprecsys source tree not found'
}

# 3. Pindah direktori dan atur PYTHONPATH
Set-Location $repoRoot
$env:PYTHONPATH = "$sourceRoot;$env:PYTHONPATH"

# 4. Kompilasi ekstensi Cython
$setupScript = Join-Path $scriptDir 'a2tcysetup.py'
& python $setupScript build_ext --inplace

if ($LASTEXITCODE -ne 0) {
    throw "a2tcysetup.py build_ext failed with exit code: $LASTEXITCODE"
}

# 5. Verifikasi modul Cython dan backend info
$verifyScript = @"
import importlib
from cooprecsys.models.ary2tower.towers import backend_info

CYTHON_MODULES = [
    '_cy_types',
    '_cy_forward',
    '_cy_predict',
    '_cy_similarity',
    '_cy_train',
]
BASE_PATH = 'cooprecsys.models.ary2tower.CLtowers.'

for mod_name in CYTHON_MODULES:
    full_module_path = BASE_PATH + mod_name
    module = importlib.import_module(full_module_path)
    print(f'[OK] {full_module_path} -> {module.__file__}')

info = backend_info()
assert info['compiled'], info
print(f'[OK] backend -> {info}')
"@

& python -c $verifyScript
if ($LASTEXITCODE -ne 0) {
    throw 'Cython backend verification failed'
}