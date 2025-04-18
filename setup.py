from setuptools import setup, find_packages

setup(
    name="momento_capital",  # Reemplaza con el nombre de tu librería
    version="0.1.2",
    author="Sergio Montes",
    author_email="ss.montes.jimenez@gmail.com",
    description="MomentoCapital Library",
    packages=find_packages(),
    install_requires=["quantstats", "matplotlib", "pandas", "numpy", "godolib"],
    python_requires=">=3.9",
)
