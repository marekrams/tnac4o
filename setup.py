"""Setup script for otn2d."""
from setuptools import setup, find_packages

setup(
    name='otn2d',
    description='otn2d',       # TODO: upgrade description
    long_description='otn2d',
    platform=['Linux', 'Unix'],
    install_requires=['numpy', 'scipy'],
    author='Marek M. Rams, Masoud Mohseni, Bartłomiej Gardas',
    packages=find_packages(exclude='examples')
)
