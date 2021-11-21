import numpy as np
import pytest

from ampute import UnivariateAmputer


@pytest.mark.parametrize(
    "params, err_type, err_msg",
    [
        ({"strategy": "random"}, ValueError, "The strategy 'random' is not supported."),
        ({"subset": [2.5, 3.5]}, TypeError, "All entry in `subset` should all be"),
        ({"subset": ["a", "b"]}, TypeError, "Passing a list of strings in `subset`"),
    ],
)
def test_univariate_amputer_validation_parameters(params, err_type, err_msg):
    rng = np.random.RandomState(42)
    X = rng.rand(10, 10)

    with pytest.raises(err_type, match=err_msg):
        UnivariateAmputer(**params).fit_transform(X)


def test_univariate_amputer_unknown_feature_names():
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(42)
    n_features = 5
    X = pd.DataFrame(
        rng.rand(10, n_features), columns=[f"Column {i}" for i in range(n_features)]
    )

    with pytest.raises(ValueError, match="Feature 'xxx' is not a feature name in X."):
        UnivariateAmputer(subset=["xxx"]).fit_transform(X)
