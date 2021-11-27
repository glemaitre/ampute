import numpy as np
import pytest

from sklearn.utils._testing import _convert_container

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
    """Validate the `UnivariateAmputer` parameters."""
    rng = np.random.RandomState(42)
    X = rng.rand(10, 2)

    with pytest.raises(err_type, match=err_msg):
        UnivariateAmputer(**params).fit_transform(X)


def test_univariate_amputer_unknown_feature_names():
    """Check the error raised with unknown feature names."""
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(42)
    n_features = 5
    X = pd.DataFrame(
        rng.rand(10, n_features), columns=[f"Column {i}" for i in range(n_features)]
    )

    with pytest.raises(ValueError, match="Feature 'xxx' is not a feature name in X."):
        UnivariateAmputer(subset=["xxx"]).fit_transform(X)


@pytest.mark.parametrize("X_type", ["array", "sparse", "dataframe", "list"])
@pytest.mark.parametrize("copy", [False, True])
def test_univariate_amputer_subset(X_type, copy):
    """Check that subset is selected a subset of features to amputate."""
    rng = np.random.RandomState(0)
    n_samples, n_features = 10_000, 5
    column_names = np.array([f"Column {i}" for i in range(n_features)], dtype=object)

    X = rng.randn(n_samples, n_features)
    X = _convert_container(X, X_type, columns_name=column_names)

    subset_idx = np.array([0, 2], dtype=np.int64)
    subset = column_names[subset_idx] if X_type == "dataframe" else subset_idx
    amputer = UnivariateAmputer(strategy="mcar", subset=subset, copy=copy)
    X_amputated = amputer.fit_transform(X)

    if X_type == "sparse":
        X_amputated = X_amputated.toarray()
    else:
        X_amputated = np.asarray(X_amputated)

    for feature_idx in range(n_features):
        col = X_amputated[:, feature_idx]
        n_missing = np.sum(np.isnan(col))
        if feature_idx in subset_idx:
            assert abs(n_missing / n_samples - 0.5) < 0.04
        else:
            assert n_missing == 0


@pytest.mark.parametrize("X_type", ["array", "sparse", "dataframe", "list"])
@pytest.mark.parametrize("ratio_missingness", [0.5, [0.2, 0.3, 0.4, 0.5, 0.6]])
@pytest.mark.parametrize("copy", [False, True])
def test_univariate_amputer_mcar(X_type, ratio_missingness, copy):
    """Check that we differently ampute columns based on the ratio."""
    rng = np.random.RandomState(0)
    n_samples, n_features = 10_000, 5
    column_names = [f"Column {i}" for i in range(n_features)]

    X = rng.randn(n_samples, n_features)
    X = _convert_container(X, X_type, columns_name=column_names)

    amputer = UnivariateAmputer(
        strategy="mcar", ratio_missingness=ratio_missingness, copy=copy
    )
    X_amputated = amputer.fit_transform(X)

    if X_type == "sparse":
        X_amputated = X_amputated.toarray()
    else:
        X_amputated = np.asarray(X_amputated)

    ratios = (
        [ratio_missingness] * n_features
        if isinstance(ratio_missingness, float)
        else ratio_missingness
    )
    for ratio, feature_idx in zip(ratios, range(n_features)):
        col = X_amputated[:, feature_idx]
        n_missing = np.sum(np.isnan(col))

        assert abs(n_missing / n_samples - ratio) < 0.04


@pytest.mark.parametrize("X_type", ["array", "sparse", "dataframe"])
@pytest.mark.parametrize("copy", [False, True])
def test_univariate_amputer_copy(X_type, copy):
    """Check that that we do the amputation inplace if requested."""
    rng = np.random.RandomState(0)
    n_samples, n_features = 10_000, 5
    column_names = [f"Column {i}" for i in range(n_features)]

    X = rng.randn(n_samples, n_features)
    X = _convert_container(X, X_type, columns_name=column_names)

    amputer = UnivariateAmputer(strategy="mcar", copy=copy)
    X_amputed = amputer.fit_transform(X)

    if not copy:
        assert X is X_amputed
    else:
        assert not (X is X_amputed)


@pytest.mark.parametrize(
    "X_type, X_amputed_type",
    [
        ("list", "ndarray"),
        ("array", "ndarray"),
        ("sparse", "csr_matrix"),
        ("dataframe", "DataFrame"),
    ],
)
def test_univariate_amputer_output_type(X_type, X_amputed_type):
    """Check the output type once amputed."""
    rng = np.random.RandomState(0)
    n_samples, n_features = 10_000, 5
    column_names = [f"Column {i}" for i in range(n_features)]

    X = rng.randn(n_samples, n_features)
    X = _convert_container(X, X_type, columns_name=column_names)

    amputer = UnivariateAmputer(strategy="mcar")
    X_amputed = amputer.fit_transform(X)

    assert type(X_amputed).__name__ == X_amputed_type
