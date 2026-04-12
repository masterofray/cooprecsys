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