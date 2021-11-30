import numpy as np
import pytest

from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import parametrize_with_checks

from ampute import RandomLogisticRegression


def test_random_logistic_regression():
    n_samples, n_features = 10_000, 10
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, random_state=0
    )

    mean_probabilities = 0.8
    model = RandomLogisticRegression(
        mean_probability=mean_probabilities, random_state=0
    )
    model.fit(X, y)

    assert model.coef_.shape == (1, n_features)
    assert model.intercept_.shape == (1,)
    np.testing.assert_array_equal(model.classes_, [0, 1])

    y_prob = model.predict_proba(X)
    np.testing.assert_allclose(y_prob.sum(axis=1), np.ones(n_samples))

    y_pred = model.predict(X)
    assert y_pred.shape == (n_samples,)

    assert model.predict_proba(X)[:, 1].mean() == pytest.approx(mean_probabilities)


@parametrize_with_checks([RandomLogisticRegression(random_state=0)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
