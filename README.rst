.. -*- mode: rst -*-

|Continuous integration|_ |Codecov|_ |Documentation|_ |Black|_

.. |Continuous integration| image:: https://github.com/glemaitre/ampute/actions/workflows/ci.yml/badge.svg?branch=main
.. _`Continuous integration`: https://github.com/glemaitre/ampute/actions/workflows/ci.yml

.. |Codecov| image:: https://codecov.io/gh/glemaitre/ampute/branch/main/graph/badge.svg?token=nnKm1BeGD3
.. _Codecov: https://codecov.io/gh/glemaitre/ampute

.. |Documentation| image:: https://readthedocs.org/projects/ampute/badge/?version=latest
.. _Documentation: https://ampute.readthedocs.io/en/latest/?badge=latest

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

ampute
======

This Python package provides some `scikit-learn` compatible utilities to
generate dataset with missing values from a given dataset.

Installation
------------

Dependencies
------------

`ampute` requires the following Python packages:

- `numpy` >= 1.17.5
- `scipy`>= 1.0.1
- `scikit-learn` >=1.0

The examples in the documentation require `pandas`, `matplotlib`, and
`seaborn`.

User Installation
-----------------

You can install `ampute` using pip::

    pip install ampute
