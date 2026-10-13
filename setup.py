from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# -ffp-contract=off: no fused multiply-add, so the C++ core stays bitwise equal to the Python
# reference. clang on arm64 fuses `v - phi * dec` by default; x86-64 builds without FMA don't.
ext = [Pybind11Extension("mboq._core", ["cpp/mboq_core.cpp"], cxx_std=20,
                         extra_compile_args=["-O3", "-ffp-contract=off"])]
setup(package_dir={"": "src"}, packages=["mboq"], ext_modules=ext, cmdclass={"build_ext": build_ext})
