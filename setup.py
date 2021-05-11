from setuptools import setup, find_packages

requirements = [
    "numpy>=1.19.5,<1.20.0",
    "scipy>=1.5.4,<1.6.0",
    "sympy>=1.5.1,<1.9.0",
]

setup(
    name="py-quantized-sapt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
)
