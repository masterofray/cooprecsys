#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-05-29"
__modified__   = "2026-08-19"

from pathlib import Path
import sys
from Cython.Build import cythonize
import numpy as np
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

BASE_DIR = Path(__file__).resolve().parent
CYTHON_DIR = BASE_DIR / "CLproximity"
SRC_DIR = BASE_DIR.parents[1]  # .../src
REPO_ROOT = BASE_DIR.parents[2]  # .../

try:
    REL_SRC_DIR = str(SRC_DIR.relative_to(Path.cwd()))
except ValueError:
    REL_SRC_DIR = str(SRC_DIR)


# Custom build_ext untuk memastikan folder inplace dibuat sebelum file .so disalin
class CustomBuildExt(build_ext):

    def copy_extensions_to_source(self):
        for ext in self.extensions:
            inplace_file = self.get_ext_fullpath(ext.name)
            Path(inplace_file).parent.mkdir(parents=True, exist_ok=True)
        super().copy_extensions_to_source()


if __name__ == "__main__":
    if sys.platform == "win32":
        extra_compile_args = ["/O2", "/openmp"]
        extra_link_args = []
        omp_lib = []
    elif sys.platform == "darwin":
        extra_compile_args = [
            "-O3",
            "-Xpreprocessor",
            "-fopenmp",
            "-ffast-math",
        ]
        extra_link_args = ["-lomp"]
        omp_lib = []
    else:
        # For Linux (WSL Ubuntu) / GCC
        extra_compile_args = ["-O3", "-fopenmp", "-ffast-math"]
        extra_link_args = ["-fopenmp"]
        omp_lib = []

    # FIX 1: Prefix modul disesuaikan dengan posisi cysetup.py terhadap subfolder CLproximity
    # Ini memastikan file .so tergenerasi tepat di dalam folder CLproximity/
    modprefix = "CLproximity"

    # FIX 2: Directori pencarian pxd & C headers
    inc_dirs = [
        np.get_include(),
        str(CYTHON_DIR),
        str(BASE_DIR),
        str(SRC_DIR),
        str(REPO_ROOT),
    ]

    pyx_files = list(CYTHON_DIR.glob("*.pyx"))
    if not pyx_files:
        raise FileNotFoundError(
            f"[ERROR] Tidak ditemukan berkas *.pyx di direktori: {CYTHON_DIR}"
        )

    exts = []
    for pyx in pyx_files:
        module_name = f"{modprefix}.{pyx.stem}"
        exts.append(
            Extension(
                name=module_name,
                sources=[str(pyx)],
                language="c",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                libraries=omp_lib,
                include_dirs=inc_dirs,
            )
        )

    setup(
        name="arycolbring",
        version="0.0.2",
        description=(
            "Ultra-optimised user-to-item collaborative filtering "
            "with Cython + OpenMP kernels"
        ),
        author="aryanto",
        python_requires=">=3.10",
        packages=find_packages(where=str(SRC_DIR)),
        package_dir={"": REL_SRC_DIR},
        cmdclass={"build_ext": CustomBuildExt},
        ext_modules=cythonize(
            exts,
            include_path=inc_dirs,  # Memaksa Cython mencari .pxd di semua inc_dirs
            compiler_directives={
                "language_level": 3,
                "boundscheck": False,
                "wraparound": False,
                "initializedcheck": False,
                "embedsignature": True,
                "cdivision": True,
            },
            annotate=False,
        ),
        zip_safe=False,
    )