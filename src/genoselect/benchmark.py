"""Benchmark several genomic-prediction models on the same data."""
from __future__ import annotations

from .crossval import CVResult, cross_validate
from .models import available_models

__all__ = ["benchmark"]


def benchmark(geno, y, models=None, k: int = 5, reps: int = 1,
              scheme: str = "kfold", groups=None, random_state=None) -> CVResult:
    """Cross-validate all (or selected) models and return a comparable summary.

    Defaults to every model except the (slow, nested) ensemble. Use
    ``CVResult.summary()`` for a tidy table sorted by predictive ability.
    """
    if models is None:
        models = [m for m in available_models() if m != "ensemble"]
    return cross_validate(geno, y, models=models, k=k, reps=reps,
                          scheme=scheme, groups=groups, random_state=random_state)
