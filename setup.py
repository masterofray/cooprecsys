#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, Masterofray/Rekomendasi Produk Koperasi"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.0.1rc2"
__maintainer__ = "Aryanto"
__email__      = "aryanto.dandan@gmail.com"
__status__     = "Development"
__created__    = "2026-06-15"

"""
Root setup.py wrapper for cooprecsys.
Delegates Cython compilation strictly to src/models/arycolbring/cysetup.py
to preserve production-grade OpenMP and compiler flags.
"""

import sys
import subprocess
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


def Discover():
    packages = ["cooprecsys"]
    root = Path("src")
    for init_file in root.rglob("__init__.py"):
        rel = init_file.parent.relative_to(root)
        if str(rel) == ".":
            continue
        pkg = "cooprecsys." + ".".join(rel.parts)
        packages.append(pkg)
    return packages


class Proxies(build_ext):
    """
    Hijacking the standard build_ext instruction to 
    delegate execution to the arycolbring 
    architecture's cysetup.py.
    """
    def run(self):
        possible_dirs = [
            Path("src/models/arycolbring").resolve(),
            Path("models/arycolbring").resolve()
        ]
        target_dir = None
        for d in possible_dirs:
            if (d / "cysetup.py").exists():
                target_dir = d
                break

        if target_dir:
            is_windows = sys.platform.startswith("win")

            if is_windows:
                script_name = "cysetup.ps1"
                cmd = ["pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_name]
            else:
                script_name = "cysetup.sh"
                cmd = ["bash", script_name]

            if not (target_dir / script_name).exists():
                raise FileNotFoundError(
                    f"Expected '{script_name}' in {target_dir} for this platform "
                    f"(sys.platform={sys.platform!r}), but it was not found."
                )

            print(f"->> Delegating Cython kernel compilation to: "
                  f"{script_name} in {target_dir}")
            subprocess.check_call(cmd, cwd=target_dir)
            fix      = Path(self.build_lib).resolve()
            dest_dir = fix / "cooprecsys" / "models" / "arycolbring" / "CLproximity"
            dest_dir.mkdir(parents = True, exist_ok = True)
            for ext in ["*.so", "*.pyd", "*.dll", "*.dylib"]:
                for file_path in target_dir.glob(ext):
                    print(f"->> Injecting compiled binary {file_path.name} into {dest_dir}")
                    target_file = dest_dir / file_path.name
                    target_file.write_bytes(file_path.read_bytes())
        else:
            print(f"WARNING: Cython Target dir is not found.")
        self.extensions = list()
        super().run()


if __name__ == '__main__':
    setup(
        package_dir  = {"cooprecsys": "src"},
        packages     = Discover(),
        package_data = {"": ["*.c", "*.so", "*.pyd", "*.dll", "*.dylib", 
                             "*.pyx", "*.pxd", "config.ini", "py.typed",
                             "*.html", "*.j2", "*.png", "*.css", "*.js",
                             "*.ico", "*.sh", "*.ps1", "*.md", "*.sql", "*.jpg",
                             ]},
        include_package_data = True,
        ext_modules          = [Extension("cooprecsys", sources=[])],
        cmdclass             = {"build_ext": Proxies},
        zip_safe             = False,
        )