#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-07-30"

import sys
import numpy as np
from pathlib import Path
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages

BASE_DIR = Path(__file__).resolve().parent
CYTHON_DIR = BASE_DIR / "CLtowers"

# Navigasi path ke src/ dan root repository
try:
    SRC_DIR = BASE_DIR.parents[1]     # .../src
    REPO_ROOT = BASE_DIR.parents[2]   # .../
except IndexError:
    SRC_DIR = BASE_DIR
    REPO_ROOT = BASE_DIR

try:
    REL_SRC_DIR = str(SRC_DIR.relative_to(Path.cwd()))
except ValueError:
    REL_SRC_DIR = str(SRC_DIR)

compiler_directives = {
    "language_level"   : 3,
    "boundscheck"      : False,
    "wraparound"       : False,
    "initializedcheck" : False,
    "embedsignature"   : True,
    "cdivision"        : True,
}

def get_cython_extensions():
    """Returns (cythonized_ext_modules, compiler_directives) for setuptools consumption."""
    if sys.platform == "win32":
        extra_compile_args = ["/O2", "/openmp"]
        extra_link_args    = []
        omp_lib            = []
    elif sys.platform == "darwin":
        extra_compile_args = ["-O3", "-fopenmp", "-march=native"]
        extra_link_args    = ["-fopenmp"]
        omp_lib            = ["omp"]
    else:
        extra_compile_args = [
            "-O3",
            "-fopenmp",
            "-march=native",
            "-ffast-math",
        ]
        extra_link_args    = ["-fopenmp"]
        omp_lib            = []

    # Penentuan nama module path secara dinamis berdasarkan posisi folder relatif terhadap src/
    try:
        rel_base = BASE_DIR.relative_to(SRC_DIR)
        pkg_prefix = ".".join(rel_base.parts) + ".CLtowers"
    except ValueError:
        pkg_prefix = "CLtowers"

    inc_dirs = [
        np.get_include(),
        str(CYTHON_DIR),
        str(BASE_DIR),
        str(SRC_DIR),
        str(REPO_ROOT),
    ]

    exts = []
    for pyx in CYTHON_DIR.glob("*.pyx"):
        module_name = f"{pkg_prefix}.{pyx.stem}"
        rel_pyx_path = pyx.relative_to(Path.cwd()).as_posix()
        exts.append(
            Extension(
                name               = module_name,
                sources            = [rel_pyx_path],
                language           = "c",
                extra_compile_args = extra_compile_args,
                extra_link_args    = extra_link_args,
                libraries          = omp_lib,
                include_dirs       = inc_dirs,
            )
        )

    return cythonize(
        exts,
        include_path=inc_dirs,
        compiler_directives=compiler_directives,
        annotate=False
    ), compiler_directives


if __name__ == '__main__':
    ext_modules, directives = get_cython_extensions()

    setup(
        name            = "ary2tower",
        version         = "0.0.1",
        description     = (
            "Two-tower neural recommender (user/item MLP towers) "
            "with Cython + OpenMP kernels"
        ),
        author          = "aryanto",
        python_requires = ">=3.8",
        packages        = find_packages(where=str(SRC_DIR)),
        package_dir     = {"": REL_SRC_DIR},
        ext_modules     = ext_modules,
        zip_safe        = False,
    )