"""
Setup script for building Dark Soliton Furnace Python extension
"""
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "dark_soliton_furnace",
        ["furnace_bindings.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "../ccl",
            # OpenCL headers - adjust path as needed
            "/usr/include",  # Linux
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include",  # Windows CUDA
            "/System/Library/Frameworks/OpenCL.framework/Headers"  # macOS
        ],
        libraries=["OpenCL"],
        library_dirs=[
            "/usr/lib/x86_64-linux-gnu",  # Linux
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64",  # Windows
            "/System/Library/Frameworks/OpenCL.framework/Libraries"  # macOS
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", "1.0.0")],
    ),
]

setup(
    name="dark_soliton_furnace",
    version="1.0.0",
    author="TORI Team",
    description="GPU-accelerated dark soliton dynamics for chaos computing",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

# Build with: python setup.py build_ext --inplace
