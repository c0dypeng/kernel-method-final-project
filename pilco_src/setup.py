from setuptools import setup, find_packages

setup(
    name="pilco",
    version="0.2",
    packages=find_packages("."),
    install_requires=[],  # all deps come from the parent Dockerfile
)
