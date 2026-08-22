#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
if [[ -d "$REPO_ROOT/src/cooprecsys" ]]; then SOURCE_ROOT="$REPO_ROOT/src"; elif [[ -d "$REPO_ROOT/cooprecsys" ]]; then SOURCE_ROOT="$REPO_ROOT"; else echo "ERROR: cooprecsys source tree not found" >&2; exit 1; fi
cd "$REPO_ROOT"
python3 "$SCRIPT_DIR/a2tcysetup.py" build_ext --inplace
PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import importlib
from cooprecsys.models.ary2tower.towers import backend_info
mods = ['_cy_types','_cy_forward','_cy_predict','_cy_similarity','_cy_train']
base='cooprecsys.models.ary2tower.CLtowers.'
for name in mods:
    m=importlib.import_module(base+name); print('[OK]', base+name, '->', m.__file__)
assert backend_info()['compiled'], backend_info()
print('[OK] backend ->', backend_info())
PY
