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


import sys
import numpy        as np
from   pathlib      import Path
from   Cython.Build import cythonize
from   setuptools   import setup, Extension, find_packages

BASE_DIR   = Path(__file__).resolve().parent
CYTHON_DIR = BASE_DIR / "CLproximity"
SRC_ROOT_DIR = BASE_DIR.parents[2]  # menunjuk ke root repository / src
if __name__ == '__main__':
    exts = list()
    if sys.platform == "win32":
        extra_compile_args = ["/O2", "/openmp"]
        extra_link_args    = list()
        omp_lib            = list()
    elif sys.platform == "darwin":
        # Homebrew LLVM clang supports -fopenmp
        extra_compile_args = ["-O3", "-fopenmp", "-ffast-math"]
        extra_link_args    = ["-fopenmp"]
        omp_lib            = ["omp"]
    else: 
        # For Unix
        extra_compile_args = ["-O3", "-fopenmp", 
                              "-ffast-math",]
        extra_link_args    = ["-fopenmp"]
        omp_lib            = list()

    modpreffix = "cooprecsys.models.arycolbring.CLproximity"
    for pyx in CYTHON_DIR.glob("*.pyx"):
        #module_name = (f"CLproximity.{pyx.stem}")
        module_name = (f"{modpreffix}.{pyx.stem}")
        #relative    = pyx.relative_to(BASE_DIR)
        exts.append(Extension(
            name               = module_name,
            sources            = [str(pyx)],
            language           = "c",
            extra_compile_args = extra_compile_args,
            extra_link_args    = extra_link_args,
            libraries          = omp_lib,
            include_dirs       = [np.get_include(), str(CYTHON_DIR), str(SRC_ROOT_DIR)]
            )
    setup(
        name            = "arycolbring",
        version         = "0.0.2",
        description     = ("Ultra-optimised user-to-item collaborative filtering "
                           "with Cython + OpenMP kernels"),
        author          = "aryanto",
        python_requires = ">=3.10",
        packages        = find_packages(),
        ext_modules     = cythonize(
                          exts,
                          compiler_directives = {
                          "language_level"   : 3,
                          "boundscheck"      : False,
                          "wraparound"       : False,
                          "initializedcheck" : False,
                          "embedsignature"   : True,
                          "cdivision"        : True},
                          annotate           = False),
        zip_safe        = False)
