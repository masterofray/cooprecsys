#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-06-15"

"""
Root setup.py wrapper for cooprecsys.
Delegates Cython compilation strictly to src/models/arycolbring/cysetup.py
to preserve production-grade OpenMP and compiler flags.
"""
import os
import sys
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

class ProxyBuildExt(build_ext):
    """
    Membajak instruksi build_ext standar untuk mendelegasikan eksekusi
    secara mutlak ke cysetup.py milik arsitektur arycolbring.
    """
    def run(self):
        possible_dirs = [
        os.path.abspath(os.path.join("src", "models", "arycolbring")),
        os.path.abspath(os.path.join("models", "arycolbring")),
        os.path.abspath(os.path.join("cooprecsys", "models", "arycolbring")),
        os.path.abspath(os.path.join("cooprecsys", "src", "models", "arycolbring"))]
        target_dir    = None
        for d in possible_dirs:
            if os.path.exists(os.path.join(d, "cysetup.py")):
                target_dir = d
                break

        script_name = "cysetup.py"
        if target_dir:
            print(f"->> Delegating Cython kernel compilation to:"
                  f"{script_name} in {target_dir}")
            subprocess.check_call(
            [sys.executable, script_name, "build_ext", "--inplace"],
            cwd = target_dir)
        else:
            print(f"WARNING: Cython Target dir {target_dir} is not found.")
        super().run()

if __name__ == '__main__':
    if os.path.exists("src/cooprecsys") or \
       os.path.exists("src/cooprecsys/__init__.py"):
        PACKAGE_DIR = {"": "src"}
        WHERE = "src"
    else:
        PACKAGE_DIR = {"": "."}
        WHERE = "."

    setup(
        name         = "cooprecsys",
        version      = "0.0.1rc",
        description  = "Koperasi Recommender System Core Engine",
        package_dir  = {"": "src"},
        packages     = find_packages(
                       where   = PACKAGE_DIR, 
                       include = ["cooprecsys*"]),
        package_data = {"": ["*.so", "*.pyd", "*.dll", "*.dylib", 
                             "*.pyx", "*.pxd", "config.ini", "py.typed"]},
        include_package_data = True,
        ext_modules          = [Extension("arycolbring_proxy", sources=[])],
        cmdclass             = {"build_ext": ProxyBuildExt},
        zip_safe             = False)
