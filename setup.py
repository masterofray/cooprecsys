from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "fpgrowth_core",
        sources=["./src/models/fpgrowth/fpgrowth_core.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        annotate=True,
        quiet=True,
    )
]

setup(
    ext_modules=cythonize(ext_modules, 
                          language_level=3,
                          annotate=True),
    zip_safe=False,
)

# Dari LTR LGBM
"""
setup.py
========
Package installation script for ltr_framework.
"""

from setuptools import setup, find_packages

setup(
    name             = "ltr_framework",
    version          = "1.0.0",
    description      = "Production-grade Learning-to-Rank pipeline built on LightGBM",
    author           = "Aryanto",
    author_email     = "aryanto.dandan@gmail.com",
    python_requires  = ">=3.10",
    packages         = find_packages(),
    install_requires = [
        "lightgbm>=4.0.0",
        "optuna>=3.0.0",
        "duckdb>=0.9.0",
        "mlflow>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pyarrow>=12.0.0",
    ],
    extras_require = {
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "ruff",
            "mypy",
        ]
    },
    entry_points = {
        "console_scripts": [
            "ltr-train=ltr_framework.main:_build_cli_parser",
        ]
    },
    classifiers = [
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
