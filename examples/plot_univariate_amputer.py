"""
=====================
Univariate amputation
=====================

This example demonstrates different ways to amputate a dataset in an univariate
manner.
"""

# Author: G. Lemaitre
# License: BSD 3 clause

# %%
import sklearn
import seaborn as sns

sklearn.set_config(display="diagram")
sns.set_context("poster")

# %%
import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10_000, n_features=10, random_state=42)

feature_names = [f"Features #{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)

# %%
import numpy as np
from ampute import UnivariateAmputer

rng = np.random.default_rng(42)

n_features_with_missing_values = 3
features_with_missing_values = rng.choice(
    feature_names, size=n_features_with_missing_values, replace=False
)
amputer = UnivariateAmputer(
    strategy="mcar",
    subset=features_with_missing_values,
    ratio_missingness=[0.2, 0.3, 0.4],
)

# %%
X_missing = amputer(X)
X_missing.head()

# %%
import matplotlib.pyplot as plt

ax = X_missing[features_with_missing_values].isna().mean().plot.barh()
ax.set_title("Proportion of missing values")
plt.tight_layout()

# %%
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

model = make_pipeline(
    amputer,
    StandardScaler(),
    SimpleImputer(strategy="mean"),
    LogisticRegression(),
)
model

# %%
n_folds = 100
cv = ShuffleSplit(n_splits=n_folds, random_state=42)
results = pd.Series(
    cross_val_score(model, X, y, cv=cv, n_jobs=2),
    index=[f"Fold #{i}" for i in range(n_folds)],
)

# %%
ax = results.plot.hist()
ax.set_xlim([0, 1])
ax.set_xlabel("Accuracy")
ax.set_title("Cross-validation scores")
plt.tight_layout()
