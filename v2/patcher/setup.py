from setuptools import setup
import pybind11
from torch.utils.cpp_extension import BuildExtension, CppExtension
import sys

include_dirs = [pybind11.get_include()]
library_dirs = []

if sys.platform == "win32":
    extra_compile_args = ['/std:c++17', '/O2', '/MD']
    extra_link_args = ['/NODEFAULTLIB:libcmt']
    include_dirs.append(r'C:\Users\anton\vcpkg\installed\x64-windows-static\include'),
    library_dirs.append(r'C:\Users\anton\vcpkg\installed\x64-windows-static\lib')
else:
    extra_compile_args = ['-std=c++17', '-O3']
    extra_link_args = []

setup(
    name="patcher",
    ext_modules=[CppExtension("patcher", ["src/main.cpp", "src/bindings.cpp", "src/context.cpp"],
                               libraries=["zstd", "torch", "c10"],
                               extra_compile_args=extra_compile_args,
                               extra_link_args=extra_link_args,
                               include_dirs=include_dirs,
                               library_dirs=library_dirs)],
    cmdclass={"build_ext": BuildExtension}
)
