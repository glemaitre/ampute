import numpy as np
import pytest

from ampute import UnivariateAmputer


@pytest.mark.parametrize(
    "params, err_type, err_msg",
    [
        ({"strategy": "random"}, ValueError, "The strategy 'random' is not supported."),
    ],
)
def test_univariate_amputer_validation_parameters(params, err_type, err_msg):
    rng = np.random.RandomState(42)
    X = rng.rand(10, 10)

    with pytest.raises(err_type, match=err_msg):
        UnivariateAmputer(**params).fit_transform(X)
