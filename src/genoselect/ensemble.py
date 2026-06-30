"""Stacked super-learner ensemble for genomic prediction."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

__all__ = ["StackedEnsemble"]

_DEFAULT_BASE = ("gblup", "elastic_net", "random_forest", "gradient_boosting")


class StackedEnsemble(BaseEstimator, RegressorMixin):
    """Super-learner that combines base models with non-negative weights.

    Out-of-fold predictions from an inner cross-validation are stacked, and the
    combination weights are found by non-negative least squares (then
    normalised to sum to one), following the super-learner of van der Laan et
    al. (2007). Base models are then refit on the full training data.
    """

    def __init__(self, base_models=None, inner_k: int = 5, random_state=None):
        self.base_models = base_models
        self.inner_k = inner_k
        self.random_state = random_state

    def fit(self, X, y):
        from .models import make_model

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        names = list(self.base_models) if self.base_models else list(_DEFAULT_BASE)

        n = y.shape[0]
        oof = np.zeros((n, len(names)))
        kf = KFold(n_splits=self.inner_k, shuffle=True,
                   random_state=self.random_state)
        for tr, te in kf.split(X):
            for j, nm in enumerate(names):
                try:
                    mod = make_model(nm, random_state=self.random_state)
                    mod.fit(X[tr], y[tr])
                    oof[te, j] = np.asarray(mod.predict(X[te])).ravel()
                except Exception:
                    oof[te, j] = y[tr].mean()

        w, _ = nnls(oof, y)
        if w.sum() == 0:
            w = np.ones(len(names))
        w = w / w.sum()
        self.weights_ = dict(zip(names, w))

        self.fitted_ = {}
        for nm in names:
            try:
                mod = make_model(nm, random_state=self.random_state)
                mod.fit(X, y)
                self.fitted_[nm] = mod
            except Exception:
                pass
        # keep only models that fitted; renormalise their weights
        self.names_ = [nm for nm in names if nm in self.fitted_]
        wsum = sum(self.weights_[nm] for nm in self.names_)
        if wsum > 0:
            self.weights_ = {nm: self.weights_[nm] / wsum for nm in self.names_}
        else:
            self.weights_ = {nm: 1.0 / len(self.names_) for nm in self.names_}
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.column_stack(
            [np.asarray(self.fitted_[nm].predict(X)).ravel() for nm in self.names_]
        )
        w = np.array([self.weights_[nm] for nm in self.names_])
        return preds @ w
