from setuptools import setup, find_packages

setup(
    name='MotifCompendium',
    version='0.2',
    packages=find_packages(),
    package_data={
        'MotifCompendium': ['utils/*'],
    },
    install_requires=[
        # List any dependencies here
        "h5py", "igraph", "jinja2", "leidenalg",
        "logomaker", "numpy", "pandas", "seaborn", "scikit-learn"
    ],
)