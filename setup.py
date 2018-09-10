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

# PyPI doesn't render markdown yet. Workaround for a sane appearance
# https://github.com/pypa/pypi-legacy/issues/148#issuecomment-227757822
README = opj(dirname(__file__), 'README.md')
try:
    import pypandoc
    long_description = pypandoc.convert(README, 'rst')
except (ImportError, OSError) as exc:
    # attempting to install pandoc via brew on OSX currently hangs and
    # pypandoc imports but throws OSError demanding pandoc
    print(
        "WARNING: pypandoc failed to import or thrown an error while converting"
        " README.md to RST: %r   .md version will be used as is" % exc
    )
    long_description = open(README).read()


setup(
    name="remodnav",
    author="The REMoDNaV Team and Contributors",
    author_email="michael.hanke@gmail.com",
    version=version,
    description="robust eye movement detection for natural viewing",
    long_description=long_description,
    packages=[pkg for pkg in find_packages('.') if pkg.startswith('remodnav')],
    install_requires=[
        'numpy',
        'scipy',
        'statsmodels',
        'matplotlib',
    ],
    extras_require={
        'devel-docs': [
            # used for converting README.md -> .rst for long_description
            'pypandoc',
        ]},
    entry_points = {
        'console_scripts': ['remodnav=remodnav:main'],
    },
)
