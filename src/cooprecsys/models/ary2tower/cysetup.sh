#!/usr/bin/env bash
set -euo pipefail

# 1. Penentuan direktori skrip dan root repositori
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

# 2. Deteksi lokasi struktur kode sumber (source tree)
if [[ -d "$REPO_ROOT/src/cooprecsys" ]]; then
  SOURCE_ROOT="$REPO_ROOT/src"
elif [[ -d "$REPO_ROOT/cooprecsys" ]]; then
  SOURCE_ROOT="$REPO_ROOT"
else
  echo "ERROR: cooprecsys source tree not found" >&2
  exit 1
fi

# 3. Kompilasi ekstensi Cython di root repositori
cd "$REPO_ROOT"
python3 "$SCRIPT_DIR/a2tcysetup.py" build_ext --inplace

# 4. Verifikasi modul Cython yang berhasil dikompilasi
PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
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
    print(f"[OK] {full_module_path} -> {module.__file__}")

info = backend_info()
assert info['compiled'], info
print(f"[OK] backend -> {info}")
PY