#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

from os.path import join as opj
from os.path import dirname


def get_version():
    """Load version without any imports
    """
    with open(opj(dirname(__file__), 'remodnav', '__init__.py')) as f:
        version_lines = list(filter(lambda x: x.startswith('__version__'), f))
    assert (len(version_lines) == 1)
    return version_lines[0].split('=')[1].strip(" '\"\t\n")


# extension version
version = get_version()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="remodnav",
    author="The REMoDNaV Team and Contributors",
    author_email="michael.hanke@gmail.com",
    version=version,
    description="robust eye movement detection for natural viewing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/psychoinformatics-de/remodnav",
    packages=[pkg for pkg in find_packages('.') if pkg.startswith('remodnav')],
    install_requires=[
        'numpy',
        'scipy',
        'statsmodels',
        'matplotlib',
    ],
    entry_points = {
        'console_scripts': ['remodnav=remodnav:main'],
    },
)
