"""Genomic relationship matrices and marker centring."""
from __future__ import annotations

import numpy as np

__all__ = ["centre_markers", "vanraden_grm", "allele_frequencies"]


def allele_frequencies(geno: np.ndarray) -> np.ndarray:
    """Reference-allele frequencies (p) from a 0/1/2 dosage matrix."""
    geno = np.asarray(geno, dtype=float)
    return np.nanmean(geno, axis=0) / 2.0


def centre_markers(geno: np.ndarray, min_maf: float = 0.0):
    """Centre markers by 2p and return the VanRaden scaling.

    Returns ``(W, p, c, keep)`` where ``W`` is the centred marker matrix
    (n x m_kept), ``p`` the allele frequencies of the kept markers, ``c`` the
    VanRaden normaliser ``2 * sum p(1-p)``, and ``keep`` the boolean column mask.
    """
    geno = np.asarray(geno, dtype=float)
    p = np.nanmean(geno, axis=0) / 2.0
    maf = np.minimum(p, 1.0 - p)
    keep = maf >= min_maf
    p = p[keep]
    W = geno[:, keep] - 2.0 * p
    c = 2.0 * np.sum(p * (1.0 - p))
    if c <= 0:
        raise ValueError("No polymorphic markers after MAF filtering.")
    return W, p, c, keep


def vanraden_grm(geno: np.ndarray, min_maf: float = 0.0) -> np.ndarray:
    """VanRaden (2008) genomic relationship matrix from 0/1/2 dosages.

    G = W Wᵀ / (2 Σ p(1-p)), with W = M - 2p the centred marker matrix.
    """
    W, _p, c, _keep = centre_markers(geno, min_maf)
    return (W @ W.T) / c
