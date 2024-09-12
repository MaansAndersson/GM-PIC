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
    install_requires=['numpy < 2.0', 'scipy < 2.0', 'seaborn < 1.0', 'scikit-learn < 2.0', 'pandas < 3.0'],
)
