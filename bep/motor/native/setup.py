"""Build the fused rollout + cost extension in place (bep/motor/native/_fused*.so).

    .venv/bin/python bep/motor/native/setup.py build_ext --inplace
(bep.motor.native.bridge builds it automatically the first time it is imported.)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco
import pybind11
from setuptools import Extension, setup

here = Path(__file__).resolve().parent
os.chdir(here)
mj = Path(mujoco.__file__).resolve().parent
lib = sorted(mj.glob("libmujoco.so.*"))[0]
ext = Extension(
    "_fused", ["fused.cpp"],
    include_dirs=[str(mj / "include"), pybind11.get_include()],
    extra_compile_args=["-O3", "-std=c++17", "-march=native", "-DNDEBUG", "-pthread"],
    extra_link_args=[str(lib), f"-Wl,-rpath,{mj}", "-pthread"],
    language="c++")
if len(sys.argv) == 1:
    sys.argv += ["build_ext", "--inplace"]
setup(name="bep_fused", ext_modules=[ext])
