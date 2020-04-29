"""Setup script for tnac4o."""
from setuptools import setup, find_packages

requirements = open('requirements.txt').readlines()

description = ('Heuristical solver for optimization problems on quasi-2d lattices employing approximate tensor network contractions.')

# README file as long_description.
long_description = open('README.md', encoding='utf-8').read()

__version__ = '0.0.9'

setup(
    name='tnac4o',
    version=__version__,
    author='Marek M. Rams, Masoud Mohseni, Bartlomiej Gardas',
    author_email='marek.rams@uj.edu.pl',
    license='Apache License 2.0',
    platform=['any'],
    python_requires=('>=3.6.0'),
    install_requires=requirements,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude='examples')
)
