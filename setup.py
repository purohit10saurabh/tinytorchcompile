import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tinytorchcompile",
    version="0.1.0",
    author="Saurabh Purohit",
    email="saurabh97purohit@gmail.com",
    license="MIT",
    keywords="torch.compile, operator fusion, compiler, torch, numpy, PyTorch 2",
    description="`torch.compile` in a nutshell, showing its main idea: operator fusion.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/purohit10saurabh/tinytorchcompile",
    py_modules=["tinytorchcompile"],
    install_requires=["numpy>=1.24"],
    extras_require={"test": ["pytest", "torch"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
