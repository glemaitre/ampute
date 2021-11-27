from collections.abc import Iterable
from numbers import Integral, Real

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state, check_scalar


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

    def fit_transform(self, X, y=None):
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
        is_dataframe = False
        if not (hasattr(X, "__array__") or sparse.issparse(X)):
            # array-like
            X = check_array(X, force_all_finite="allow-nan", copy=self.copy, dtype=None)
        elif hasattr(X, "loc"):
            is_dataframe = True

        if self.copy:
            X = X.copy()

        n_samples, n_features = X.shape

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
            feature_names = X.columns.tolist() if is_dataframe else None
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
            ratio_missingness = np.asarray(self.ratio_missingness, dtype=np.float64)
            for ratio in ratio_missingness:
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
            ratio_missingness = np.full_like(
                self.amputated_features_indices_,
                fill_value=self.ratio_missingness,
                dtype=np.float64,
            )

        random_state = check_random_state(self.random_state)

        if self.strategy == "mcar":
            for ratio, feature_idx in zip(
                ratio_missingness, self.amputated_features_indices_
            ):
                mask_missing_values = random_state.choice(
                    [False, True], size=n_samples, p=[1 - ratio, ratio]
                )
                if is_dataframe:
                    X.iloc[mask_missing_values, feature_idx] = np.nan
                else:
                    X[mask_missing_values, feature_idx] = np.nan

        return X
