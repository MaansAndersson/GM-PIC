from setuptools import setup, find_packages


setup(
    name="GM-PIC",
    version='0.0.2',
    author="Måns Andersson",
    author_email="mansande@kth.se",
    description="Analyzing Gaussian Mixture Models as a means to compress data from Particle-in-Cell (PIC) Plasma Simulations this work is a part of the Plasma-PEPSC project",
    url="https://github.com/MaansAndersson/GM-PIC",
    install_requires=['numpy <= 3.0', 'scipy <= 3.0', 'seaborn <= 1.0', 'scikit-learn >= 1.5.1', 'pandas <= 3.0'],
    packages=find_packages(where='.',
                          include=['*','GM_PIC*','GM_PIC/mixture/*']),
    include_package_data=True,
    python_requires=">=3.8",
)

