from setuptools import setup, find_packages


setup(
    name="GM-PIC",
    version='0.0.2',
    author="Måns Andersson",
    author_email="mansande@kth.se",
    description="Analyzing Gaussian Mixture Models as a means to compress data from Particle-in-Cell (PIC) Plasma Simulations this work is a part of the Plasma-PEPSC project",
    url="https://github.com/MaansAndersson/GM-PIC",
    install_requires=['numpy <= 2.0', 'scipy <= 2.0', 'seaborn <= 1.0', 'scikit-learn <= 2.0', 'pandas <= 3.0'],
    packages=find_packages(where='GM_PIC',
                          include=['GM_PIC*']),
    include_package_data=True,
    python_requires=">=3.8",
)
