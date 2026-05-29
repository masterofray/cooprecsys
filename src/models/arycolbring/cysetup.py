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


import numpy        as np
from   pathlib      import Path
from   Cython.Build import cythonize
from   setuptools   import setup, Extension, find_packages

BASE_DIR   = Path(__file__).resolve().parent
CYTHON_DIR = BASE_DIR / "CLproximity"

if __name__ == '__main__':
    exts = list()
    for pyx in CYTHON_DIR.glob("*.pyx"):
        module_name = (f"models.arycolbring.CLproximity.{pyx.stem}")
        exts.append(Extension(
            name               = module_name,
            sources            = [str(pyx)],
            language           = "c",
            extra_link_args    = ["-fopenmp"],
            extra_compile_args = ["-O3", "-fopenmp"],
            include_dirs       = [np.get_include(), str(CYTHON_DIR)])
            )
    setup(
        name        = "arycolbring",
        version     = "0.0.1",
        packages    = find_packages(),
        ext_modules = cythonize(
                      exts,
                      compiler_directives = {
                      "language_level"   : 3,
                      "boundscheck"      : False,
                      "wraparound"       : False,
                      "initializedcheck" : False,
                      "cdivision"        : True},
                      annotate           = False),
        zip_safe    = False)
