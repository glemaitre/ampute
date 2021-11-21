from sklearn.base import BaseEstimator, TransformerMixin


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
        Whether to perform the amputation inplace or to trigger a copy.
    """

    def __init__(self, strategy="mcar", subset=None, ratio_missingness=0.5, copy=True):
        self.strategy = strategy
        self.subset = subset
        self.ratio_missingness = ratio_missingness
        self.copy = copy

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
        X_amputed : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            The dataset with missing values.
        """
        supported_strategies = ["mcar"]
        if self.strategy not in supported_strategies:
            raise ValueError(
                f"The strategy '{self.strategy}' is not supported. "
                f"Supported strategies are: {supported_strategies}"
            )

        return X
