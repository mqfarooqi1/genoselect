"""Quality control and imputation for marker matrices."""
from __future__ import annotations

import numpy as np

__all__ = ["qc_markers", "impute_markers"]


def impute_markers(geno: np.ndarray) -> np.ndarray:
    """Mean-impute missing genotypes (NaN) column-wise (by 2p)."""
    geno = np.asarray(geno, dtype=float).copy()
    if not np.isnan(geno).any():
        return geno
    col_mean = np.nanmean(geno, axis=0)
    rows, cols = np.where(np.isnan(geno))
    geno[rows, cols] = np.take(col_mean, cols)
    return geno


def qc_markers(geno: np.ndarray, maf: float = 0.05, max_missing: float = 0.1,
               impute: bool = True):
    """Filter markers by minor-allele frequency and missingness.

    Returns ``(filtered_geno, keep_mask)``. With ``impute=True`` remaining
    missing values are mean-imputed.
    """
    geno = np.asarray(geno, dtype=float)
    missing_rate = np.mean(np.isnan(geno), axis=0)
    p = np.nanmean(geno, axis=0) / 2.0
    maf_obs = np.minimum(p, 1.0 - p)
    keep = (missing_rate <= max_missing) & (maf_obs >= maf)
    out = geno[:, keep]
    if impute:
        out = impute_markers(out)
    return out, keep
