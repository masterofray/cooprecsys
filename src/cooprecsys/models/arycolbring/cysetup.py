#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GPL-3.0-only"
__version__    = "0.0.2"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"

from pathlib import Path
import sys
from Cython.Build import cythonize
import numpy as np
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

BASE_DIR = Path(__file__).resolve().parent
CYTHON_DIR = BASE_DIR / "CLproximity"

# FIXED: Pencarian direktori 'src' secara dinamis
SRC_DIR = next((p for p in BASE_DIR.parents if p.name == "src"), BASE_DIR.parents[2])
REPO_ROOT = SRC_DIR.parent

try:
    REL_SRC_DIR = str(SRC_DIR.relative_to(Path.cwd()))
except ValueError:
    REL_SRC_DIR = str(SRC_DIR)


class CustomBuildExt(build_ext):
    """Custom build_ext untuk memastikan folder tujuan dibuat sebelum file .so/.pyd disalin."""
    def copy_extensions_to_source(self):
        for ext in self.extensions:
            inplace_file = self.get_ext_fullpath(ext.name)
            Path(inplace_file).parent.mkdir(parents=True, exist_ok=True)
        super().copy_extensions_to_source()


def get_cython_extensions():
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
        extra_compile_args = ["-O3", "-fopenmp", "-ffast-math"]
        extra_link_args = ["-fopenmp"]
        omp_lib = []

    # FIXED: Menghasilkan prefix 'cooprecsys.models.arycolbring.CLproximity'
    try:
        rel_path = BASE_DIR.relative_to(SRC_DIR)
        modprefix = ".".join(rel_path.parts) + ".CLproximity"
    except ValueError:
        modprefix = "cooprecsys.models.arycolbring.CLproximity"

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
        rel_pyx_path = pyx.relative_to(Path.cwd()).as_posix()
        exts.append(
            Extension(
                name=module_name,
                sources=[rel_pyx_path],
                language="c",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                libraries=omp_lib,
                include_dirs=inc_dirs,
            )
        )

    ext_modules = cythonize(
        exts,
        include_path=inc_dirs,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "embedsignature": True,
            "cdivision": True,
        },
        annotate=False,
    )

    return ext_modules, inc_dirs


if __name__ == "__main__":
    ext_modules, inc_dirs = get_cython_extensions()

    setup(
        name="cooprecsys",
        version="0.0.2",
        description="Ultra-optimised user-to-item collaborative filtering",
        author="aryanto",
        python_requires=">=3.10",
        packages=find_packages(where=str(SRC_DIR)),
        package_dir={"": REL_SRC_DIR},
        cmdclass={"build_ext": CustomBuildExt},
        ext_modules=ext_modules,
        zip_safe=False,
    )