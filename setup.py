import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="opt_einsum_fx",
    version="0.1.1",
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
)
