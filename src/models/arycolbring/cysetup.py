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
from setuptools.command.build_ext import build_ext
BASE_DIR   = Path(__file__).resolve().parent
CYTHON_DIR = BASE_DIR / "CLproximity"
SRC_DIR = BASE_DIR.parents[1]  # .../src
REPO_ROOT = BASE_DIR.parents[2]  # .../
REL_BASE_DIR = BASE_DIR.relative_to(Path.cwd())

# Custom build_ext untuk memastikan folder inplace dibuat sebelum file .so disalin
class CustomBuildExt(build_ext):
    def copy_extensions_to_source(self):
        for ext in self.extensions:
            inplace_file = self.get_ext_fullpath(ext.name)
            Path(inplace_file).parent.mkdir(parents=True, exist_ok=True)
        super().copy_extensions_to_source()

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
    exts = []

    # Include dirs agar Cython & C-compiler bisa nemu .pxd & .h lokal
    inc_dirs = [
        np.get_include(),
        str(CYTHON_DIR),
        str(BASE_DIR),
        str(SRC_DIR),
        str(REPO_ROOT),
    ]
    for pyx in CYTHON_DIR.glob("*.pyx"):
        #module_name = (f"CLproximity.{pyx.stem}")
        module_name = f"{modpreffix}.{pyx.stem}"
        #relative    = pyx.relative_to(BASE_DIR)
        exts.append(Extension(
            name               = module_name,
            sources            = [str(pyx)],
            language           = "c",
            extra_compile_args = extra_compile_args,
            extra_link_args    = extra_link_args,
            libraries          = omp_lib,
            include_dirs=inc_dirs,
            ))
    setup(
        name            = "arycolbring",
        version         = "0.0.2",
        description     = ("Ultra-optimised user-to-item collaborative filtering "
                           "with Cython + OpenMP kernels"),
        author          = "aryanto",
        python_requires = ">=3.10",
        packages        = find_packages(),
        #PENTING: Petakan namespace ke lokasi folder fisik di disk!
        package_dir={"": str(SRC_DIR.relative_to(Path.cwd()))}
        if SRC_DIR.exists() and SRC_DIR.is_relative_to(Path.cwd())
        else {},
        cmdclass        = {"build_ext": CustomBuildExt},
        ext_modules     = cythonize(
                          exts,
                          include_path=inc_dirs,
                          compiler_directives = {
                          "language_level"   : 3,
                          "boundscheck"      : False,
                          "wraparound"       : False,
                          "initializedcheck" : False,
                          "embedsignature"   : True,
                          "cdivision"        : True},
                          annotate           = False),
        zip_safe        = False)
