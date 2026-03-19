from setuptools import setup, find_packages

setup(
    name="gan-cyber-range-sim",
    version="1.0.0",
    packages=find_packages(exclude=["tests*"]),
    install_requires=["torch>=2.0.0", "numpy>=1.24.0"],
    python_requires=">=3.9",
)
