"""GBLUP by REML (Endelman 2011 spectral method), scikit-learn compatible."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, RegressorMixin

from .relationship import centre_markers

__all__ = ["GBLUP", "gblup", "reml_solve"]


def reml_solve(y, K, X=None):
    """Solve the mixed model y = Xb + g + e, g ~ N(0, K s2u), e ~ N(0, I s2e).

    Variance components are estimated by REML using the spectral approach of
    Endelman (2011) — the profiled restricted log-likelihood is minimised over
    the single ratio lambda = s2e / s2u. Returns a dict with Vu, Ve, h2, beta,
    gebv, lambda, Hinv and the residuals (y - Xb).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.shape[0]
    if X is None:
        X = np.ones((n, 1))
    X = np.asarray(X, dtype=float)
    p = np.linalg.matrix_rank(X)

    S = np.eye(n) - X @ np.linalg.pinv(X.T @ X) @ X.T
    offset = np.sqrt(n)
    Hb = K + offset * np.eye(n)
    SHbS = S @ Hb @ S
    SHbS = (SHbS + SHbS.T) / 2.0

    vals, vecs = np.linalg.eigh(SHbS)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    phi = vals[: n - p]
    U = vecs[:, : n - p]
    theta = np.maximum(phi - offset, 0.0)
    omega_sq = (U.T @ y) ** 2
    df = n - p

    def neg_reml(log_lambda):
        lam = np.exp(log_lambda)
        d = theta + lam
        return df * np.log(np.sum(omega_sq / d)) + np.sum(np.log(d))

    res = minimize_scalar(neg_reml, bounds=(np.log(1e-9), np.log(1e9)),
                          method="bounded", options={"xatol": 1e-9})
    lam = float(np.exp(res.x))

    Vu = float(np.sum(omega_sq / (theta + lam)) / df)
    Ve = float(lam * Vu)
    h2 = Vu / (Vu + Ve) if (Vu + Ve) > 0 else float("nan")

    Hinv = np.linalg.inv(K + lam * np.eye(n))
    XtHinv = X.T @ Hinv
    beta = np.linalg.solve(XtHinv @ X, XtHinv @ y)
    resid = y - X @ beta
    gebv = K @ Hinv @ resid

    return dict(Vu=Vu, Ve=Ve, h2=h2, beta=beta, gebv=gebv,
                lam=lam, Hinv=Hinv, resid=resid)


class GBLUP(BaseEstimator, RegressorMixin):
    """Genomic BLUP regressor.

    Fits GBLUP on a 0/1/2 marker matrix, estimating variance components by REML.
    The equivalent ridge-regression marker effects are derived so the model can
    predict breeding values for unseen genotypes.

    Attributes (set after ``fit``)
    ------------------------------
    h2_ : float
        Estimated genomic heritability Vu / (Vu + Ve).
    Vu_, Ve_, lambda_ : float
        Genetic and residual variances and their ratio.
    gebv_ : ndarray
        Genomic estimated breeding values for the training individuals.
    marker_effects_ : ndarray
        RR-BLUP marker effects (for out-of-sample prediction).
    """

    def __init__(self, min_maf: float = 0.0):
        self.min_maf = min_maf

    def fit(self, X, y):
        geno = np.asarray(X, dtype=float)
        W, p, c, keep = centre_markers(geno, self.min_maf)
        K = (W @ W.T) / c
        sol = reml_solve(y, K)

        self.K_ = K
        self.Vu_, self.Ve_, self.h2_ = sol["Vu"], sol["Ve"], sol["h2"]
        self.lambda_ = sol["lam"]
        self.beta_ = sol["beta"]
        self.gebv_ = sol["gebv"]
        self.intercept_ = float(sol["beta"][0])
        self.marker_means_ = 2.0 * p
        self.marker_effects_ = (W.T @ sol["Hinv"] @ sol["resid"]) / c
        self._keep = keep
        self.n_features_in_ = geno.shape[1]
        return self

    def predict(self, X):
        geno = np.asarray(X, dtype=float)
        Wn = geno[:, self._keep] - self.marker_means_
        return self.intercept_ + Wn @ self.marker_effects_


def gblup(y, geno=None, K=None, X=None, min_maf: float = 0.0):
    """Functional interface mirroring the R ``GSbench::gblup``.

    With ``geno`` it returns a fitted :class:`GBLUP` (which can predict new
    genotypes); with only ``K`` it returns the REML solution dict (no
    out-of-sample prediction).
    """
    if geno is not None:
        return GBLUP(min_maf=min_maf).fit(geno, y)
    if K is not None:
        return reml_solve(y, np.asarray(K, dtype=float), X)
    raise ValueError("Provide either `geno` or `K`.")
