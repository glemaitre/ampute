.. -*- mode: rst -*-

|Continuous integration|_ |Codecov|_ |Black|_

.. |Continuous integration| image:: https://github.com/glemaitre/ampute/actions/workflows/ci.yml/badge.svg?branch=main
.. _`Continuous integration`: https://github.com/glemaitre/ampute/actions/workflows/ci.yml

.. |Codecov| image:: https://codecov.io/gh/glemaitre/ampute/branch/main/graph/badge.svg?token=nnKm1BeGD3
.. _Codecov: https://codecov.io/gh/glemaitre/ampute

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

- `numpy` >= 1.13.3
- `scipy`>= 0.19.1
- `scikit-learn` >=1.0

The examples in the documentation require `pandas`, `matplotlib`, and
`seaborn`.

User Installation
-----------------

You can install `ampute` using pip::

    pip install ampute
