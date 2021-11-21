#! /usr/bin/env python
"""Package to manage amputation for data science simulation purpose."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join("ampute", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "ampute"
DESCRIPTION = "Package to manage amputation for data science simulation purpose."
with codecs.open("README.rst", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "G. Lemaitre"
MAINTAINER_EMAIL = "g.lemaitre58@gmail.com"
URL = "https://github.com/glemaitre/ampute"
LICENSE = "new BSD"
DOWNLOAD_URL = "https://github.com/glemaitre/ampute"
VERSION = __version__
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: C",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]
INSTALL_REQUIRES = [
    "numpy>=1.13.3",
    "scipy>=0.19.1",
]
EXTRAS_REQUIRE = {
    "dev": [
        "black",
        "flake8",
    ],
    "tests": [
        "pytest",
        "pytest-cov",
    ],
    "docs": [
        "sphinx",
        "sphinx-gallery",
        "pydata-sphinx-theme",
        "sphinxcontrib-bibtex",
        "numpydoc",
        "matplotlib",
        "pandas",
        "seaborn",
    ],
}
EXTRAS_REQUIRE["all"] = [
    req for reqs in EXTRAS_REQUIRE.values() for req in reqs
] + INSTALL_REQUIRES


setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
