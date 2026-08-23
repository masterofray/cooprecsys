#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CYTHON_DIR = BASE_DIR / 'CLtowers'

# Support both <repo>/cooprecsys and <repo>/src/cooprecsys layouts.
candidates = []
for parent in [BASE_DIR, *BASE_DIR.parents]:
    for source_root in (parent, parent / 'src'):
        if (source_root / 'cooprecsys').is_dir():
            candidates.append(source_root.resolve())
if not candidates:
    raise RuntimeError(
        f'Cannot locate cooprecsys package root from {BASE_DIR}; '
        'expected <repo>/cooprecsys or <repo>/src/cooprecsys.'
    )
SOURCE_ROOT = candidates[0]
REPO_ROOT = SOURCE_ROOT.parent
PACKAGE_ROOT = SOURCE_ROOT / 'cooprecsys'
PACKAGE_NAME = 'cooprecsys.models.ary2tower.CLtowers'

class CustomBuildExt(build_ext):
    def copy_extensions_to_source(self):
        for ext in self.extensions:
            Path(self.get_ext_fullpath(ext.name)).parent.mkdir(parents=True, exist_ok=True)
        super().copy_extensions_to_source()

def _compiler_flags():
    if sys.platform == 'win32':
        return ['/O2', '/openmp'], [], []
    if sys.platform == 'darwin':
        return ['-O3', '-Xpreprocessor', '-fopenmp', '-ffast-math'], ['-lomp'], []
    return ['-O3', '-fopenmp', '-ffast-math'], ['-fopenmp'], []

COMPILER_DIRECTIVES = {
    'language_level': 3, 'boundscheck': False, 'wraparound': False,
    'initializedcheck': False, 'embedsignature': True, 'cdivision': True,
}

def get_cython_extensions():
    extra_compile_args, extra_link_args, libraries = _compiler_flags()
    include_dirs = [str(CYTHON_DIR), str(BASE_DIR), str(PACKAGE_ROOT), str(SOURCE_ROOT), str(REPO_ROOT), np.get_include()]
    pyx_files = sorted(CYTHON_DIR.glob('*.pyx'))
    if not pyx_files:
        raise RuntimeError(f'No Cython sources found in {CYTHON_DIR}')
    extensions = [Extension(
        name=f'{PACKAGE_NAME}.{pyx.stem}', sources=[str(pyx)], language='c',
        include_dirs=include_dirs, extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args, libraries=libraries,
    ) for pyx in pyx_files]
    return cythonize(extensions, include_path=include_dirs,
                     compiler_directives=COMPILER_DIRECTIVES,
                     annotate=False, language_level=3)

if __name__ == '__main__':
    setup(
        name='cooprecsys', version='0.0.2',
        description='Two-tower neural recommender with Cython + OpenMP kernels',
        python_requires='>=3.10',
        packages=find_packages(where=str(SOURCE_ROOT)),
        package_dir={'': str(SOURCE_ROOT)},
        cmdclass={'build_ext': CustomBuildExt},
        ext_modules=get_cython_extensions(), zip_safe=False,
    )
