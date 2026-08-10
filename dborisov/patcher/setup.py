from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="patcher",
    ext_modules=[CppExtension("patcher", ["src/main.cpp", "src/bindings.cpp", "src/context.cpp"],
                               libraries=["zstd", "torch", "c10"],
                               extra_compile_args=["-std=c++17", "-O3"])],
    cmdclass={"build_ext": BuildExtension}
)
