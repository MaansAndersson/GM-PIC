from setuptools import setup, find_packages

from GM_PIC import version

setup(
    name="GM-PIC",
    version=version.__version__,
    author="Måns Andersson",
    author_email="mansande@kth.se",
    description="Analyzing Gaussian Mixture Models as a means to compress data from Particle-in-Cell (PIC) Plasma Simulations this work is a part of the Plasma-PEPSC project",
    url="https://github.com/MaansAndersson/GM-PIC",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
)
