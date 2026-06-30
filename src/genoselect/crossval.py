"""Breeding-relevant cross-validation and honest accuracy reporting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneGroupOut

from .models import make_model

__all__ = ["cross_validate", "CVResult"]


def _accuracy(pred, obs) -> float:
    pred = np.asarray(pred, dtype=float).ravel()
    obs = np.asarray(obs, dtype=float).ravel()
    if np.std(pred) == 0 or np.std(obs) == 0:
        return float("nan")
    return float(np.corrcoef(pred, obs)[0, 1])


class CVResult:
    """Per-fold predictive abilities for one or more models."""

    def __init__(self, scores: dict[str, list[float]], scheme: str, k: int, reps: int):
        self.scores = scores
        self.scheme = scheme
        self.k = k
        self.reps = reps

    def summary(self) -> pd.DataFrame:
        rows = []
        for model, vals in self.scores.items():
            v = np.asarray(vals, dtype=float)
            rows.append(dict(
                model=model,
                mean=float(np.nanmean(v)),
                sd=float(np.nanstd(v, ddof=1)) if np.sum(~np.isnan(v)) > 1 else float("nan"),
                n_folds=int(np.sum(~np.isnan(v))),
            ))
        return (pd.DataFrame(rows)
                .sort_values("mean", ascending=False, ignore_index=True))

    def __repr__(self) -> str:
        head = f"<CVResult: {self.scheme}, {self.k}-fold x {self.reps} rep(s)>\n"
        return head + self.summary().to_string(index=False) + \
            "\n(accuracy = predictive ability, cor(pred, observed) on held-out data)"


def cross_validate(geno, y, models=None, k: int = 5, reps: int = 1,
                   scheme: str = "kfold", groups=None, random_state=None) -> CVResult:
    """Cross-validate genomic-prediction models.

    Parameters
    ----------
    geno : array (n x m)
        Marker matrix coded 0/1/2.
    y : array (n,)
        Phenotypes.
    models : str | list[str] | None
        Model name(s); defaults to ``["gblup"]``. See :func:`available_models`.
    k : int
        Number of folds for ``scheme="kfold"``.
    scheme : {"kfold", "leave_group_out"}
        ``leave_group_out`` requires ``groups`` (e.g. families/environments).
    """
    geno = np.asarray(geno, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if models is None:
        models = ["gblup"]
    if isinstance(models, str):
        models = [models]
    scores: dict[str, list[float]] = {m: [] for m in models}

    for rep in range(reps):
        if scheme == "kfold":
            rs = None if random_state is None else random_state + rep
            splits = KFold(n_splits=k, shuffle=True, random_state=rs).split(geno)
        elif scheme == "leave_group_out":
            if groups is None:
                raise ValueError("`groups` is required for leave_group_out.")
            splits = LeaveOneGroupOut().split(geno, y, groups)
        else:
            raise ValueError(f"Unknown scheme {scheme!r}.")

        for train, test in splits:
            for m in models:
                try:
                    mod = make_model(m, random_state=random_state)
                    mod.fit(geno[train], y[train])
                    pred = mod.predict(geno[test])
                    scores[m].append(_accuracy(pred, y[test]))
                except Exception:
                    scores[m].append(float("nan"))

    return CVResult(scores, scheme=scheme, k=k, reps=reps)
