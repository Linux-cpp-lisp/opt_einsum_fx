import setuptools
from distutils.spawn import find_executable
import glob
import os
import subprocess
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# == For C++ extension ==
import torch
from torch.utils.cpp_extension import (
    CUDAExtension,
    CUDA_HOME,
    BuildExtension,
)

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None

if WITH_CUDA:
    include_dirs = []
    library_dirs = []

    cuda_nvcc = find_executable("nvcc")
    cuda_root = os.path.join(os.path.dirname(cuda_nvcc), os.pardir)
    cuda_version = re.search(
        r"release ([^,]*),",
        subprocess.check_output([cuda_nvcc, "--version"]).decode("utf-8"),
    ).group(1)
    include_dirs.append(os.path.join(cuda_root, "include"))
    library_dirs.append(os.path.join(cuda_root, "lib64"))

    root = None
    if "CUTENSOR_ROOT" in os.environ:
        root = os.environ["CUTENSOR_ROOT"]
    elif "CONDA_PREFIX" in os.environ:
        root = os.environ["CONDA_PREFIX"]

    if root is not None:
        include_dirs.append(os.path.join(root, "include"))
        library_dirs.append(os.path.join(root, "lib"))
        versioned_path = os.path.join(root, "lib", cuda_version)
        if not os.path.exists(versioned_path):
            versioned_path = os.path.join(root, "lib", cuda_version.split(".")[0])
        library_dirs.append(versioned_path)

    # Determine whether to use TensorFloat32
    # This option controls matmul in pytorch, which is used internally
    # in tensordot, and thus is appropriate to use for einsum
    # Note that a seperate flag exists in PyTorch for cuDNN
    # See https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    use_tf32 = torch.backends.cuda.matmul.allow_tf32

    extensions = [
        CUDAExtension(
            name="_opt_einsum_fx",
            sources=glob.glob("src/*.cpp"),
            libraries=["cutensor"],
            define_macros=[
                ("_GLIBCXX_USE_CXX11_ABI", str(int(torch._C._GLIBCXX_USE_CXX11_ABI))),
            ]
            + ([("CUTENSOR_USE_TF32", "true")] if use_tf32 else []),
            extra_compile_args=["-std=c++14", "-fopenmp"],
            extra_link_args=["-std=c++14", "-fopenmp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=library_dirs,
        )
    ]
else:
    extensions = []

# == end extension ==

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
    install_requires=["torch>=1.8.0", "opt_einsum", "packaging"],
    packages=["opt_einsum_fx"],
    ext_modules=extensions,
    # cmdclass={"build_ext": BuildExtension},
)
