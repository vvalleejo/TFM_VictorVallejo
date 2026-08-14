from setuptools import setup, find_packages

setup(
    name="jepa_wm",
    version="0.1.0",
    description="Latent World Models for Multivariate Data using JEPA-based Architectures",
    author="Victor Vallejo",
    packages=find_packages(include=["jepa_wm", "jepa_wm.*"]),
    python_requires=">=3.9",
)
