.. _getting_started:

###############
Getting Started
###############

Prerequisites
=============

The `ampute` package requires the following dependencies:

* python (>=3.7)
* numpy (>=1.17.5)
* scipy (>=1.0.1)
* scikit-learn (>=1.0)

Install
=======

From PyPi or conda-forge repositories
-------------------------------------

`ampute` is currently available on the PyPi's repositories and you can install
it via `pip`::

  pip install -U ampute

The package is release also in `conda-forge` platform::

  conda install -c conda-forge ampute

From source available on GitHub
-------------------------------

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from Github and install all dependencies::

  git clone https://github.com/glemaitre/ampute.git
  cd ampute
  pip install .

Be aware that you can install in developer mode with::

  pip install --no-build-isolation --editable .

If you wish to make pull-requests on GitHub, we advise you to install
pre-commit::

  pip install pre-commit
  pre-commit install

Test and coverage
=================

You want to test the code before to install::

  $ make test

You wish to test the coverage of your version::

  $ make coverage

You can also use `pytest`::

  $ pytest ampute -v

Contribute
==========

You can contribute to this code through Pull Request on GitHub_. Please, make
sure that your code is coming with unit tests to ensure full coverage and
continuous integration in the API.

.. _GitHub: https://github.com/glemaitre/ampute/pulls
