'''
Create by Aryanto
at 20260323
email me : aryanto.dandan@gmail.com
'''

"""
Setup script for Cython compilation
Run: python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "cyltr.feature_engineering",
        ["cyltr/feature_engineering.pyx"],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=['-O3', '-march=native']
    ),
    Extension(
        "cyltr.model_trainer",
        ["cyltr/model_trainer.pyx"],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=['-O3', '-march=native']
    ),
    Extension(
        "cyltr.predictor",
        ["cyltr/predictor.pyx"],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=['-O3', '-march=native']
    ),
]

setup(
    name="xgboost_ltr_ranker",
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()],
)