"""genoselect: scikit-learn-compatible genomic prediction and selection.

A unified, Pythonic toolkit for genomic prediction from SNP marker data:
the VanRaden genomic relationship matrix, GBLUP by REML, machine-learning
predictors, a stacked super-learner ensemble, breeding-relevant
cross-validation, simulation, and quality control.
"""
from __future__ import annotations

from .benchmark import benchmark
from .crossval import CVResult, cross_validate
from .ensemble import StackedEnsemble
from .gblup import GBLUP, gblup, reml_solve
from .models import available_models, make_model
from .qc import impute_markers, qc_markers
from .relationship import allele_frequencies, centre_markers, vanraden_grm
from .simulate import Population, simulate_population

__version__ = "0.1.0"

__all__ = [
    "GBLUP", "gblup", "reml_solve",
    "vanraden_grm", "centre_markers", "allele_frequencies",
    "available_models", "make_model",
    "cross_validate", "CVResult", "benchmark",
    "StackedEnsemble",
    "simulate_population", "Population",
    "qc_markers", "impute_markers",
    "__version__",
]
