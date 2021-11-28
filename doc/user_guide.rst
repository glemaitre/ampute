.. title:: User guide: contents

.. _user_guide:

==========
User Guide
==========

Learning a predictive model in presence of missing data is a difficult task.
This topic is still under investigation in research. In this regard, these
investigations required to study different type of missingness mechanisms in
a control environment.

The `ampute` package provides a set of tools to amputate a given dataset. It
allows to handcraft the missingness mechanism and pattern that later be
studied.

What are missing values?
------------------------

Let's first define a couple of notation that we will use in this documentation
Our dataset is denoted by :math:`X` of shape :math:`(n, p)` where :math:`n` is
the number of samples and :math:`p` is the number of features.
Programmatically, we can represent such a matrix as a NumPy array::

    >>> from numpy.random import default_rng
    >>> rng = default_rng(0)
    >>> n_samples, n_features = 5, 3
    >>> X = rng.normal(size=(n_samples, n_features))
    >>> X
    array([[ 0.12573022, -0.13210486,  0.64042265],
           [ 0.10490012, -0.53566937,  0.36159505],
           [ 1.30400005,  0.94708096, -0.70373524],
           [-1.26542147, -0.62327446,  0.04132598],
           [-2.32503077, -0.21879166, -1.24591095]])

If we are lucky, for each sample (i.e. row of :math:`X`), we always have an
observation. However, we could be unlucky -e.g. one of the sensor collecting
data was broken- and some observations could be missing.

In NumPy, the sentinel generally used to represent missing values is `np.nan`.
For instance, if for the first sample in :math:`X`, the value of the second
feature is not collected, we should have::

    >>> import numpy as np
    >>> X_missing = X.copy()
    >>> X_missing[0, 1] = np.nan
    >>> X_missing
    array([[ 0.12573022,         nan,  0.64042265],
           [ 0.10490012, -0.53566937,  0.36159505],
           [ 1.30400005,  0.94708096, -0.70373524],
           [-1.26542147, -0.62327446,  0.04132598],
           [-2.32503077, -0.21879166, -1.24591095]])

In a machine learning setting where one is interested about training a
predictive model, such missing values are problematic. Let's define a linear
relationship (with some noise) between the data matrix and the target vector
that we want to predict::

    >>> coef = rng.normal(size=n_features)
    >>> y = X @ coef + rng.normal(scale=0.1, size=n_samples)
    >>> y
    array([-0.18157161,  0.2046067 , -1.26059589,  1.38942449,  2.14918582])

If we would like to train a linear regression on this problem, we could use
for instance `scikit-learn`::

    >>> from sklearn.linear_model import LinearRegression
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.coef_
    array([-0.67735802, -0.70333477, -0.32484547])

However, with the presence of missing values, we cannot use this approach::

    >>> try:
    ...     reg.fit(X_missing, y)
    ... except ValueError as e:
    ...     print(e)
    Input X contains NaN.
    LinearRegression does not accept missing values encoded as NaN natively.
    For supervised learning, you might want to consider
    sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept
    missing values encoded as NaNs natively. Alternatively, it is possible to
    preprocess the data, for instance by using an imputer transformer in a pipeline
    or drop samples with missing values.
    See https://scikit-learn.org/stable/modules/impute.html

`scikit-learn` is giving you a straightforward answer informing you that it
does not accept missing values. Thus, one needs to a strategy to deal with
missing values.

In the following section, we describe what are the missingness mechanisms used
in the literature to simulate the presence of missing values in a dataset.

Missingness mechanisms
----------------------

In the literature, there are three reported mechanism to control the
missingness of a dataset: missing completely at random (MCAR), missing
at random (MAR), and missing not at random (MNAR). We will give a brief
description of each of them.

MCAR strategy is straightforward. For a given feature, missing values are
created at random. Let's imagine the first feature in :math:`X` contains
missing values. MCAR strategy is be equivalent to::

    >>> X_missing_mcar = X.copy()
    >>> missing_values_indices = rng.choice(
    ...    n_samples, size=n_samples // 2, replace=False
    ... )
    >>> X_missing_mcar[missing_values_indices, 0] = np.nan
    >>> X_missing_mcar
    array([[ 0.12573022, -0.13210486,  0.64042265],
           [ 0.10490012, -0.53566937,  0.36159505],
           [        nan,  0.94708096, -0.70373524],
           [        nan, -0.62327446,  0.04132598],
           [-2.32503077, -0.21879166, -1.24591095]])

Here, there is not link between the missing values and the features in
:math:`X`.
