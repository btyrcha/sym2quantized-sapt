from setuptools import setup, find_packages

setup(
    name="py-quantized-sapt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
