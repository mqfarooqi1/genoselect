"""Model registry: a common interface to genomic-prediction estimators."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .gblup import GBLUP

__all__ = ["available_models", "make_model"]

_MODELS = ("gblup", "elastic_net", "random_forest", "gradient_boosting", "ensemble")


def available_models():
    """Names of the genomic-prediction models genoselect can run."""
    return list(_MODELS)


def make_model(name: str, random_state=None):
    """Construct a fresh, unfitted estimator by name.

    All estimators follow the scikit-learn ``fit(X, y)`` / ``predict(X)`` API,
    where ``X`` is a 0/1/2 marker matrix.
    """
    key = name.lower()
    if key == "gblup":
        return GBLUP()
    if key == "elastic_net":
        return make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5,
                         max_iter=10000, random_state=random_state),
        )
    if key == "random_forest":
        return RandomForestRegressor(n_estimators=500, n_jobs=1,
                                     random_state=random_state)
    if key == "gradient_boosting":
        return HistGradientBoostingRegressor(random_state=random_state)
    if key == "ensemble":
        from .ensemble import StackedEnsemble
        return StackedEnsemble(random_state=random_state)
    raise ValueError(f"Unknown model {name!r}. Choose from {available_models()}.")
