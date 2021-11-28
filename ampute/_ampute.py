from collections.abc import Iterable
from numbers import Integral, Real

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state, check_scalar
from sklearn.utils.validation import _num_features, _num_samples


class UnivariateAmputer(TransformerMixin, BaseEstimator):
    """Ampute a datasets in an univariate manner.

    Univariate imputation refer to the introduction of missing values, one
    feature at a time.

    Parameters
    ----------
    strategy : str, default="mcar"
        The missingness strategy to ampute the data. Possible choices are:

        - `"mcar"`: missing completely at random. This strategy implies that
          the missing values are amputated for a feature without any dependency
          with other features.

    subset : list of {int, str}, default=None
        The subset of the features to be amputated. If None, all features are
        amputated.

    ratio_missingness : float or array-like, default=0.5
        The ratio representing the amount of missing data to be generated.
        If a float, all features to be imputed will have the same ratio.
        If an array-like, the ratio of missingness for each feature will be
        drawn from the array. It should be consistent with `subset` when a list
        is provided for `subset`.

    copy : bool, default=True
        Whether to perform the amputation inplace or to trigger a copy. The
        default will trigger a copy.

    Attributes
    ----------
    amputated_features_indices_ : ndarray of shape (n_selected_features,)
        The indices of the features that have been amputated.

    Examples
    --------
    >>> from numpy.random import default_rng
    >>> rng = default_rng(0)
    >>> n_samples, n_features = 5, 3
    >>> X = rng.normal(size=(n_samples, n_features))

    One can amputate values using the common transformer `scikit-learn` API:

    >>> amputer = UnivariateAmputer(random_state=42)
    >>> amputer.fit_transform(X)
    array([[ 0.12573022, -0.13210486,  0.64042265],
           [        nan, -0.53566937,         nan],
           [        nan,         nan,         nan],
           [        nan,         nan,  0.04132598],
           [-2.32503077,         nan, -1.24591095]])

    The amputer can be used in a scikit-learn :class:`~sklearn.pipeline.Pipeline`.

    >>> from sklearn.impute import SimpleImputer
    >>> from sklearn.pipeline import make_pipeline
    >>> pipeline = make_pipeline(
    ...     UnivariateAmputer(random_state=42),
    ...     SimpleImputer(strategy="mean"),
    ... )
    >>> pipeline.fit_transform(X)
    array([[ 0.12573022, -0.13210486,  0.64042265],
           [-1.09965028, -0.53566937, -0.18805411],
           [-1.09965028, -0.33388712, -0.18805411],
           [-1.09965028, -0.33388712,  0.04132598],
           [-2.32503077, -0.33388712, -1.24591095]])

    You can use the class as a callable if you don't need to use a
    :class:`sklearn.pipeline.Pipeline`:

    >>> from ampute import UnivariateAmputer
    >>> UnivariateAmputer(random_state=42)(X)
    array([[ 0.12573022, -0.13210486,  0.64042265],
        [        nan, -0.53566937,         nan],
        [        nan,         nan,         nan],
        [        nan,         nan,  0.04132598],
        [-2.32503077,         nan, -1.24591095]])
    """

    def __init__(
        self,
        strategy="mcar",
        subset=None,
        ratio_missingness=0.5,
        copy=True,
        random_state=None,
    ):
        self.strategy = strategy
        self.subset = subset
        self.ratio_missingness = ratio_missingness
        self.copy = copy
        self.random_state = random_state

    def fit(self, X, y=None):
        """Validation of the parameters of amputer.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape (n_samples, n_features)
            The dataset to be amputated.

        y : Ignored
            Present to follow the scikit-learn API.

        Returns
        -------
        self
            The validated amputer.
        """
        n_features = _num_features(X)

        supported_strategies = ["mcar"]
        if self.strategy not in supported_strategies:
            raise ValueError(
                f"The strategy '{self.strategy}' is not supported. "
                f"Supported strategies are: {supported_strategies}"
            )

        def convert_feature(fx, feature_names):
            """Convert feature names into positional indices."""
            if isinstance(fx, str):
                if feature_names is None:
                    raise TypeError(
                        "Passing a list of strings in `subset` is only supported "
                        "when X is a pandas DataFrame."
                    )
                try:
                    fx = feature_names.index(fx)
                    return fx
                except ValueError as e:
                    raise ValueError(
                        f"Feature '{fx}' is not a feature name in X."
                    ) from e
            elif isinstance(fx, Integral):
                return int(fx)
            else:
                raise TypeError(
                    "All entry in `subset` should all be strings or integers."
                )

        if self.subset is not None:
            feature_names = X.columns.tolist() if hasattr(X, "columns") else None
            self.amputated_features_indices_ = np.array(
                [
                    convert_feature(fx, feature_names=feature_names)
                    for fx in self.subset
                ],
                dtype=np.int64,
            )
        else:
            self.amputated_features_indices_ = np.arange(n_features, dtype=np.int64)

        if isinstance(self.ratio_missingness, Iterable):
            if len(self.ratio_missingness) != len(self.amputated_features_indices_):
                raise ValueError(
                    "The length of `ratio_missingness` should be equal to the "
                    f"length of `subset`. Pass an array-like with {n_features} "
                    "elements."
                )
            self._ratio_missingness = np.asarray(
                self.ratio_missingness, dtype=np.float64
            )
            for ratio in self._ratio_missingness:
                check_scalar(
                    ratio,
                    "ratio_missingness",
                    Real,
                    min_val=0.0,
                    max_val=1.0,
                    include_boundaries="neither",
                )
        else:
            check_scalar(
                self.ratio_missingness,
                "ratio_missingness",
                Real,
                min_val=0.0,
                max_val=1.0,
                include_boundaries="neither",
            )
            self._ratio_missingness = np.full_like(
                self.amputated_features_indices_,
                fill_value=self.ratio_missingness,
                dtype=np.float64,
            )

        return self

    def transform(self, X, y=None):
        """Amputate the dataset `X` with missing values.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape (n_samples, n_features)
            The dataset to be amputated.

        y : Ignored
            Present to follow the scikit-learn API.

        Returns
        -------
        X_amputed : {ndarray, sparse matrix, dataframe} of shape (n_samples, n_features)
            The dataset with missing values.
        """
        n_samples = _num_samples(X)

        is_dataframe = False
        if not (hasattr(X, "__array__") or sparse.issparse(X)):
            # array-like
            X = check_array(X, force_all_finite="allow-nan", copy=self.copy, dtype=None)
        elif hasattr(X, "loc"):
            is_dataframe = True

        if self.copy:
            X = X.copy()

        random_state = check_random_state(self.random_state)

        if self.strategy == "mcar":
            for ratio, feature_idx in zip(
                self._ratio_missingness, self.amputated_features_indices_
            ):
                mask_missing_values = random_state.choice(
                    [False, True], size=n_samples, p=[1 - ratio, ratio]
                )
                if is_dataframe:
                    X.iloc[mask_missing_values, feature_idx] = np.nan
                else:
                    X[mask_missing_values, feature_idx] = np.nan

        return X

    def __call__(self, X):
        """Callable that is a shorthand for calling `fit_transform`.

        This callable is useful if you don't want to integrate the transformer
        into a pipeline and impute directly a dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape (n_samples, n_features)
            The dataset to be amputated.

        Returns
        -------
        X_amputed : {ndarray, sparse matrix, dataframe} of shape (n_samples, n_features)
            The dataset with missing values.

        Examples
        --------
        >>> from numpy.random import default_rng
        >>> rng = default_rng(0)
        >>> n_samples, n_features = 5, 3
        >>> X = rng.normal(size=(n_samples, n_features))

        You can use the class as a callable if you don't need to use a
        :class:`sklearn.pipeline.Pipeline`:

        >>> from ampute import UnivariateAmputer
        >>> UnivariateAmputer(random_state=42)(X)
        array([[ 0.12573022, -0.13210486,  0.64042265],
           [        nan, -0.53566937,         nan],
           [        nan,         nan,         nan],
           [        nan,         nan,  0.04132598],
           [-2.32503077,         nan, -1.24591095]])
        """
        return self.fit_transform(X)
