import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# For C++ extension
import torch
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    BuildExtension,
)

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None

extensions = [
    (CUDAExtension if WITH_CUDA else CppExtension)(
        name="_opt_einsum_fx",
        sources=["src/tensordot.cpp"],
        extra_compile_args=[],
        language="c++",
    )
]

setuptools.setup(
    name="opt_einsum_fx",
    version="0.1.2",
    author="Linux-cpp-lisp",
    url="https://github.com/Linux-cpp-lisp/opt_einsum_fx",
    description="Einsum optimization using opt_einsum and PyTorch FX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    license_files="LICENSE",
    project_urls={
        "Bug Tracker": "https://github.com/Linux-cpp-lisp/opt_einsum_fx/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.6",
    install_requires=["torch>=1.8.0", "opt_einsum"],
    packages=["opt_einsum_fx"],
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension},
)
