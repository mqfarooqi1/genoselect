"""Single-marker genome-wide association (GWAS) with structure correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["gwas", "manhattan", "qq_plot"]


def gwas(geno, y, marker_map=None, n_pc=0, covariates=None):
    """Single-marker additive association scan.

    Each marker is tested for association with ``y`` after regressing out an
    intercept, optional ``covariates``, and (optionally) the first ``n_pc``
    genotype principal components to correct for population structure.

    Parameters
    ----------
    geno : array (n, m)
        Marker matrix coded 0/1/2.
    y : array (n,)
        Phenotype (missing values are dropped).
    marker_map : DataFrame or dict, optional
        Per-marker annotation; recognised columns ``marker``, ``chrom``, ``pos``
        are carried through to the result (enabling position-aware Manhattan
        plots).
    n_pc : int
        Number of genotype PCs to include as covariates.
    covariates : array (n, k), optional
        Extra fixed covariates.

    Returns
    -------
    pandas.DataFrame
        One row per marker with columns ``index`` [, ``marker``, ``chrom``,
        ``pos``], ``effect``, ``se``, ``t``, ``p`` and ``log10p``.
    """
    geno = np.asarray(geno, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    keep = ~np.isnan(y)
    geno, y = geno[keep], y[keep]
    n, m = geno.shape

    cols = [np.ones(n)]
    if covariates is not None:
        cov = np.asarray(covariates, dtype=float)
        if cov.shape[0] == keep.size:
            cov = cov[keep]
        cols.append(cov.reshape(n, -1))
    if n_pc > 0:
        U, S, _ = np.linalg.svd(geno - geno.mean(0), full_matrices=False)
        cols.append((U * S)[:, :n_pc])
    C = np.column_stack(cols)

    Cpinv = np.linalg.pinv(C)
    ry = y - C @ (Cpinv @ y)
    RM = geno - C @ (Cpinv @ geno)
    dfree = n - C.shape[1] - 1

    ss = (RM ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (RM * ry[:, None]).sum(0) / ss
        resid_var = ((ry[:, None] - RM * beta) ** 2).sum(0) / dfree
        se = np.sqrt(resid_var / ss)
        tval = beta / se
    pval = 2 * stats.t.sf(np.abs(tval), dfree)

    data = {"index": np.arange(m)}
    if marker_map is not None:
        mm = pd.DataFrame(marker_map).reset_index(drop=True)
        for c in ("marker", "chrom", "pos"):
            if c in mm.columns:
                data[c] = mm[c].to_numpy()
    data.update({"effect": beta, "se": se, "t": tval, "p": pval,
                 "log10p": -np.log10(pval)})
    return pd.DataFrame(data)


def _chrom_order(values):
    uniq = list(dict.fromkeys(values))
    return sorted(uniq, key=lambda c: (not str(c).isdigit(),
                                       int(c) if str(c).isdigit() else str(c)))


def manhattan(result, alpha=0.05, ax=None, colors=("#2c7fb8", "#8fbcd4"),
              title="GWAS Manhattan"):
    """Manhattan plot from a :func:`gwas` result (position-aware if ``chrom``)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("manhattan() requires matplotlib: pip install genoselect[plot]") from e

    df = result.reset_index(drop=True)
    m = len(df)
    y = df["log10p"].to_numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    if "chrom" in df.columns:
        ch = df["chrom"].astype(str).to_numpy()
        pos = df["pos"].to_numpy() if "pos" in df.columns else None
        x = np.zeros(m); cidx = np.zeros(m, dtype=int); ticks = []; off = 0.0
        gap = max(1.0, m * 0.01)
        for i, c in enumerate(_chrom_order(ch)):
            sel = np.where(ch == c)[0]
            pp = (pos[sel] - pos[sel].min()) if pos is not None else np.arange(len(sel))
            x[sel] = off + pp; cidx[sel] = i % 2
            ticks.append((off + (pp.mean() if len(pp) else 0), c))
            off += (pp.max() if len(pp) else 0) + gap
        for k in (0, 1):
            s = cidx == k
            ax.scatter(x[s], y[s], s=8, color=colors[k])
        ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks])
        ax.set_xlabel("chromosome")
    else:
        ax.scatter(df["index"], y, s=8, color=colors[0])
        ax.set_xlabel("marker index")

    thr = -np.log10(alpha / m)
    ax.axhline(thr, ls="--", color="#d95f02", label=f"Bonferroni (alpha={alpha})")
    ax.set_ylabel(r"$-\log_{10}(p)$"); ax.set_title(title); ax.legend(loc="upper right")
    return ax


def qq_plot(result, ax=None, title="QQ plot"):
    """QQ plot of observed vs expected -log10(p) from a :func:`gwas` result."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("qq_plot() requires matplotlib: pip install genoselect[plot]") from e
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    o = np.sort(result["log10p"].to_numpy()[np.isfinite(result["log10p"])])[::-1]
    e = -np.log10((np.arange(1, len(o) + 1) - 0.5) / len(o))
    ax.scatter(e, o, s=8, color="#1b9e77")
    ax.plot([0, e.max()], [0, e.max()], color="grey")
    ax.set(title=title, xlabel=r"expected $-\log_{10}(p)$", ylabel=r"observed $-\log_{10}(p)$")
    return ax
