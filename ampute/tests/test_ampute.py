import numpy as np
import pytest

from ampute import UnivariateAmputer


@pytest.mark.parametrize(
    "params, err_type, err_msg",
    [
        ({"strategy": "random"}, ValueError, "The strategy 'random' is not supported."),
        ({"subset": [2.5, 3.5]}, TypeError, "All entry in `subset` should all be"),
        ({"subset": ["a", "b"]}, TypeError, "Passing a list of strings in `subset`"),
        (
            {"ratio_missingness": np.complex64(1)},
            TypeError,
            "ratio_missingness must be an instance of",
        ),
        (
            {"ratio_missingness": 1.5},
            ValueError,
            "ratio_missingness == 1.5, must be < 1.0.",
        ),
        (
            {"ratio_missingness": [0.5]},
            ValueError,
            "The length of `ratio_missingness` should be equal",
        ),
        (
            {"ratio_missingness": [1.5, 1.5]},
            ValueError,
            "ratio_missingness == 1.5, must be < 1.0.",
        ),
    ],
)
def test_univariate_amputer_validation_parameters(params, err_type, err_msg):
    rng = np.random.RandomState(42)
    X = rng.rand(10, 2)

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
