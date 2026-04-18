"""Setup for Cython compilation of collaborative filtering modules."""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "collaborative_filtering",
        ["collaborative_filtering.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-march=native"],
        extra_link_args=["-O3"],
    )
]

setup(
    name="collaborative_filtering",
    ext_modules=cythonize(
        extensions,
        language_level="3",
        compiler_directives={"boundscheck": False, "wraparound": False}
    ),
    include_dirs=[np.get_include()]
)