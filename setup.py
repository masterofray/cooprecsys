#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GPL-3.0-only"
__version__    = "0.0.2"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"

import importlib.util
import sys
from pathlib import Path
from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

# 1. Cari lokasi cysetup.py secara dinamis
possible_cysetup_paths = [
    SRC_DIR / "cooprecsys" / "models" / "arycolbring" / "cysetup.py",
    SRC_DIR / "models" / "arycolbring" / "cysetup.py",
]

cysetup_path = next((p for p in possible_cysetup_paths if p.exists()), None)

ext_modules = []

# 2. Import cysetup.py secara native Python tanpa subprocess/shell script
if cysetup_path:
    spec = importlib.util.spec_from_file_location("cysetup_module", cysetup_path)
    if spec and spec.loader:
        cysetup_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cysetup_module)

        if hasattr(cysetup_module, "get_cython_extensions"):
            ext_modules, _ = cysetup_module.get_cython_extensions()
            print(f"->> Successfully loaded {len(ext_modules)} Cython extensions from {cysetup_path.name}")
        else:
            print(f"WARNING: 'get_cython_extensions()' not found in {cysetup_path}")
else:
    print("WARNING: Could not locate 'cysetup.py'. Building pure Python wheel without C extensions.")


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