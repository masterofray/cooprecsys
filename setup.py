#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GPL-3.0-only"
__version__    = "0.0.3"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"

import importlib.util
import sys
from pathlib import Path
from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

# 1. Daftar lokasi potensial untuk cysetup (arycolbring & ary2tower)
possible_cysetup_paths = [
    # Modul arycolbring
    SRC_DIR / "cooprecsys" / "models" / "arycolbring" / "cysetup.py",
    SRC_DIR / "models" / "arycolbring" / "cysetup.py",
    # Modul ary2tower
    SRC_DIR / "cooprecsys" / "models" / "ary2tower" / "a2tcysetup.py",
    SRC_DIR / "models" / "ary2tower" / "a2tcysetup.py",
]

# Deteksi file yang ada tanpa duplikasi
active_setup_paths = []
seen_paths = set()
for path in possible_cysetup_paths:
    if path.exists() and path.resolve() not in seen_paths:
        seen_paths.add(path.resolve())
        active_setup_paths.append(path)

ext_modules = []

# 2. Load ekstensi Cython dari semua modul yang terdeteksi secara native
for setup_path in active_setup_paths:
    module_alias = f"_cysetup_{setup_path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_alias, setup_path)
    
    if spec and spec.loader:
        try:
            cysetup_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cysetup_module)

            if hasattr(cysetup_module, "get_cython_extensions"):
                loaded_exts, _ = cysetup_module.get_cython_extensions()
                ext_modules.extend(loaded_exts)
                rel_path = setup_path.relative_to(ROOT_DIR)
                print(f"->> Successfully loaded {len(loaded_exts)} Cython extension(s) from {rel_path}")
            else:
                print(f"WARNING: 'get_cython_extensions()' not found in {setup_path}")
        except Exception as err:
            print(f"ERROR: Failed to load Cython setup from {setup_path}: {err}")

if not ext_modules:
    print("WARNING: Could not locate active Cython setups. Building pure Python wheel without C extensions.")


# 3. Eksekusi setup standar PEP 517 / setuptools
if __name__ == "__main__":
    setup(
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        package_data={
            "": [
                "*.c",
                "*.so",
                "*.pyd",
                "*.dll",
                "*.dylib",
                "*.pyx",
                "*.pxd",
                "*.ini",
                "py.typed",
                "*.html",
                "*.j2",
                "*.png",
                "*.css",
                "*.js",
                "*.ico",
                "*.md",
                "*.sql",
                "*.jpg",
            ]
        },
        include_package_data=True,
        ext_modules=ext_modules,
        zip_safe=False,
    )