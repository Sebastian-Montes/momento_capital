from setuptools import setup, find_packages

setup(
    name="momento_capital",  # Reemplaza con el nombre de tu librería
    version="0.3.9",
    author="Sergio Montes",
    author_email="ss.montes.jimenez@gmail.com",
    description="MomentoCapital Library",
    packages=find_packages(),
    install_requires=["matplotlib", "pandas", "numpy", "scipy", "xlsxwriter"],
    python_requires=">=3.9",
)
