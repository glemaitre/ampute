import numpy as np
from scipy.optimize import fsolve
from scipy.special import expit

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets, type_of_target


class RandomLogisticRegression(LogisticRegression):
    """Binary random logistic regression.

    The weights are randomly assigned. The intercept is fitted such that the
    mean probability is set to a given value corresponding to the expected
    percentage of missing values.

    Parameters
    ----------
    mean_probability : float, default=0.5
        The expected value of the probability of being missing.

    random_state : int or RandomState, default=None
        The seed of the pseudo random number generator.
        This random number generator is used to randomly geneare the weights.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features)
        The coefficients of the logistic regression model.

    intercept_ : ndarray of shape (1,)
        The intercept of the logistic regression model.
    """

    def __init__(self, mean_probability=0.5, random_state=None):
        super().__init__(random_state=random_state)
        self.mean_probability = mean_probability

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used for the purpose of this model.

        Returns
        -------
        self
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=True,
            y_numeric=False,
            ensure_min_samples=2,
        )
        check_classification_targets(y)
        target_type = type_of_target(y)
        if target_type != "binary":
            raise ValueError(
                f"{self.__class__.__name__} only supports binary targets. Got "
                f"type '{target_type}' instead."
            )
        self.classes_ = LabelEncoder().fit(y).classes_
        if self.classes_.size != 2:
            raise ValueError(
                f"A single class was provided at `fit` time, but the model "
                f"expects a binary classification problem. Got {self.classes_}."
            )
        self.coef_ = random_state.randn(1, X.shape[1])
        # normalize coefficients to have a unit variance without scaling
        # the data
        self.coef_ /= np.std((X @ self.coef_.T), axis=0)

        def fit_intercept(intercept):
            return np.mean(expit(X @ self.coef_.T + intercept) - self.mean_probability)

        intercept, info, _, _ = fsolve(fit_intercept, x0=0, full_output=True)
        self.intercept_ = np.atleast_1d(intercept)
        self.n_iter_ = info["nfev"]

        return self

    def _more_tags(self):
        return {"X_types": ["2darray"], "binary_only": True, "poor_score": True}
