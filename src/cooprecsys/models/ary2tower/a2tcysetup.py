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
from setuptools.command.build_ext import build_ext

BASE_DIR = Path(__file__).resolve().parent
CYTHON_DIR = BASE_DIR / "CLtowers"

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


compiler_directives = {
    "language_level": 3,
    "boundscheck": False,
    "wraparound": False,
    "initializedcheck": False,
    "embedsignature": True,
    "cdivision": True,
}

def get_cython_extensions():
    if sys.platform == "win32":
        extra_compile_args = ["/O2", "/openmp"]
        extra_link_args = []
        omp_lib = []
    elif sys.platform == "darwin":
        # FIXED: Menghapus -march=native agar build tidak crash di CI/CD runner
        extra_compile_args = ["-O3", "-Xpreprocessor", "-fopenmp", "-ffast-math"]
        extra_link_args = ["-lomp"]
        omp_lib = []
    else:
        # FIXED: Menghapus -march=native agar kompatibel dengan runner x86_64 GitHub Actions
        extra_compile_args = ["-O3", "-fopenmp", "-ffast-math"]
        extra_link_args = ["-fopenmp"]
        omp_lib = []

    # FIXED: Menghasilkan prefix 'cooprecsys.models.ary2tower.CLtowers'
    try:
        rel_base = BASE_DIR.relative_to(SRC_DIR)
        pkg_prefix = ".".join(rel_base.parts) + ".CLtowers"
    except ValueError:
        pkg_prefix = "cooprecsys.models.ary2tower.CLtowers"

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
                name=module_name,
                #sources=[rel_pyx_path],
                sources=[str(pyx)],
                language="c",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                libraries=omp_lib,
                include_dirs=inc_dirs,
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
        name="cooprecsys",
        version="0.0.2",
        description="Two-tower neural recommender with Cython + OpenMP kernels",
        author="aryanto",
        python_requires=">=3.10",
        packages=find_packages(where=str(SRC_DIR)),
        package_dir={"": REL_SRC_DIR},
        cmdclass={"build_ext": CustomBuildExt},
        ext_modules=ext_modules,
        zip_safe=False,
    )