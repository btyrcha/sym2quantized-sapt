from setuptools import find_packages, setup

requirements = [
    "numpy>=1.19.5,<1.27.0",
    "scipy>=1.5.4,<1.12.0",
    "sympy>=1.5.1,<1.13.0",
]

setup(
    name="py-quantized-sapt",
    version="0.1.0",
    author="Bartosz Tyrcha",
    author_email="bartektyrcha123@gmail.com",
    maintainer="Filip Brzek",
    maintainer_email="filip.brzek@gmail.com",
    description="sympy based package for second quantized operator algebra in SAPT",
    long_description=(
        "Collection of extensions to sympy second-quant module, that "
        "allows generating of linked part of expectation value of an "
        " operator within SAPT framework, "
        "using extension of Wick theorem for a case of product of fermi vacuum."
    ),
    long_description_content_type="text/x-rst",
    license="BSD-3-Clause",
    url="https://github.com/btyrcha/sym2quantized-sapt",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">3.8,<3.13",
    zip_safe=True,
)
