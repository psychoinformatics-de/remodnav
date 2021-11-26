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


try:
    import pypandoc
    README = opj(dirname(__file__), 'README.md')
    long_description = pypandoc.convert_file(README, 'rst')
except (ImportError, OSError) as exc:
    print(
        "WARNING: pypandoc failed to import or threw an error while "
        "converting README.md to RST: "
        "%r  .md version will be used as is" % exc
    )
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
    extras_require={
        'devel-docs': [
            # for converting README.md -> .rst for long description
            'pypandoc',
        ]},
    entry_points = {
        'console_scripts': ['remodnav=remodnav:main'],
    },
)
