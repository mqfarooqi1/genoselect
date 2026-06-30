"""Simulate a structured population with a heritable quantitative trait."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Population", "simulate_population"]


@dataclass
class Population:
    """A simulated population. ``geno`` is 0/1/2; ``bv`` are true breeding values."""
    geno: np.ndarray
    pheno: np.ndarray
    bv: np.ndarray
    qtl: np.ndarray
    effects: np.ndarray
    h2: float


def simulate_population(n: int = 200, m: int = 1000, n_qtl: int = 50,
                        h2: float = 0.5, seed=None) -> Population:
    """Simulate ``n`` individuals genotyped at ``m`` biallelic markers.

    ``n_qtl`` markers are causal with Normal effects; the genetic values are
    scaled to unit variance and Normal noise is added to give the target
    narrow-sense heritability ``h2``.
    """
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, size=m)
    geno = rng.binomial(2, p, size=(n, m)).astype(float)

    qtl = np.sort(rng.choice(m, size=n_qtl, replace=False))
    effects = rng.normal(size=n_qtl)
    centred = geno[:, qtl] - 2.0 * p[qtl]
    bv = centred @ effects
    sd = np.std(bv)
    if sd > 0:
        bv = bv / sd
    ve = (1.0 - h2) / h2
    pheno = bv + rng.normal(scale=np.sqrt(ve), size=n)

    return Population(geno=geno, pheno=pheno, bv=bv, qtl=qtl,
                     effects=effects, h2=h2)
