from setuptools import setup, find_packages

setup(
    name='MotifCompendium',
    version='0.1',
    packages=find_packages(),
    package_data={
        'MotifCompendium': ['utils/*'],
    },
    install_requires=[
        # List any dependencies here
    ],
)

