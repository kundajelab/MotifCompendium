from setuptools import setup, find_packages

setup(
    name="MotifCompendium",
    version="0.3",
    packages=find_packages(),
    package_data={
        "MotifCompendium": ["utils/*"],
    },
    install_requires=[
        # List any dependencies here
        "h5py",
        "igraph",
        "jinja2",
        "leidenalg",
        "logomaker",
        "numpy",
        "pandas",
        "tables",
        "seaborn",
        "scikit-learn",
        "upsetplot",
    ],
)
