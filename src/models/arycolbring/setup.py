# setup.py
"""
Build script for arycolbring Cython extensions.

Each Cython module is compiled as a separate shared library so they can
be debugged independently.  All modules are compiled with:
  - OpenMP for prange parallelism
  - aggressive optimisation flags
  - boundscheck / wraparound disabled at the compiler level

Usage
-----
    # Compile in-place (development)
    python setup.py build_ext --inplace

    # Install
    pip install .

Requirements
------------
  - Cython >= 0.29 or 3.x
  - A C compiler with OpenMP support:
      Linux  : GCC  (gcc -fopenmp)
      macOS  : clang via Homebrew  (brew install llvm; CC=clang-XX)
      Windows: MSVC (/openmp)
"""

import os
import sys
import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

# ── compiler flags ────────────────────────────────────────────────────────────
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/openmp"]
    extra_link_args    = []
    omp_lib            = []
elif sys.platform == "darwin":
    # Homebrew LLVM clang supports -fopenmp; Apple clang does not.
    extra_compile_args = ["-O3", "-fopenmp", "-march=native"]
    extra_link_args    = ["-fopenmp"]
    omp_lib            = ["omp"]
else:  # Linux
    extra_compile_args = [
        "-O3",
        "-fopenmp",
        "-march=native",
        "-ffast-math",
    ]
    extra_link_args = ["-fopenmp"]
    omp_lib         = []

include_dirs = [np.get_include()]
cy_dir       = os.path.join("arycolbring", "cy")


def cy_ext(name: str, sources=None) -> Extension:
    """
    Create an Extension for a Cython module inside ``arycolbring/cy/``.

    Parameters
    ----------
    name    : dotted module name, e.g. ``"arycolbring.cy._cy_types"``
    sources : list of .pyx paths; defaults to the single .pyx matching *name*
    """
    if sources is None:
        module_file = name.split(".")[-1] + ".pyx"
        sources     = [os.path.join(cy_dir, module_file)]
    return Extension(
        name,
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=omp_lib,
        language="c",
    )


# ── extension modules ─────────────────────────────────────────────────────────
# Order matters: shared types first so the .pxd is on the include path
# when downstream modules are compiled.

extensions = [
    cy_ext("arycolbring.cy._cy_types"),
    cy_ext("arycolbring.cy._cy_math"),
    cy_ext("arycolbring.cy._cy_representation"),
    cy_ext("arycolbring.cy._cy_update"),
    cy_ext("arycolbring.cy._cy_regularize"),
    cy_ext("arycolbring.cy._cy_fit_logistic"),
    cy_ext("arycolbring.cy._cy_fit_warp"),
    cy_ext("arycolbring.cy._cy_fit_bpr"),
    cy_ext("arycolbring.cy._cy_fit_warp_kos"),
    cy_ext("arycolbring.cy._cy_predict"),
    cy_ext("arycolbring.cy._cy_evaluate"),
]

# ── Cython compiler directives ────────────────────────────────────────────────
compiler_directives = {
    "boundscheck":      False,
    "wraparound":       False,
    "cdivision":        True,
    "initializedcheck": False,
    "nonecheck":        False,
    "embedsignature":   True,    # Allows introspection of compiled functions
    "language_level":   "3",
}

# ── setup ─────────────────────────────────────────────────────────────────────
setup(
    name="arycolbring",
    version="0.1.0",
    description=(
        "Ultra-optimised user-to-item collaborative filtering "
        "with Cython + OpenMP kernels"
    ),
    author="aryanto",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests*"]),
    package_data={
        "arycolbring":    ["config.ini"],
        "arycolbring.cy": ["*.pxd", "*.pyx"],
    },
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=True,      # produces _cy_*.html annotation files for profiling
        nthreads=4,         # parallel Cython transpilation (not OpenMP)
    ),
    include_dirs=include_dirs,
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "pandas>=1.3",
        "duckdb>=0.8",
        "tqdm>=4.60",
        "joblib>=1.1",
        "cython>=0.29",
        "seaborn>=0.12",
    ],
    zip_safe=False,
)
