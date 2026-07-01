import numpy as np
import matplotlib
matplotlib.use("Agg")

import genoselect as gs


def test_gwas_output_and_power():
    pop = gs.simulate_population(n=200, m=500, n_qtl=10, h2=0.6, seed=0)
    res = gs.gwas(pop.geno, pop.pheno, n_pc=2)
    assert len(res) == 500
    assert {"index", "effect", "se", "t", "p", "log10p"} <= set(res.columns)
    assert res["p"].between(0, 1).all()
    # causal markers should be enriched for signal
    causal = np.zeros(500, bool); causal[pop.qtl] = True
    assert res["log10p"][causal].mean() > res["log10p"][~causal].mean()
    top5 = res.sort_values("log10p", ascending=False)["index"].head(5).to_numpy()
    assert len(set(top5) & set(pop.qtl)) >= 1


def test_gwas_with_marker_map_and_plots():
    pop = gs.simulate_population(n=120, m=200, n_qtl=8, h2=0.5, seed=1)
    mmap = {"chrom": np.repeat([1, 2], 100), "pos": np.tile(np.arange(100), 2)}
    res = gs.gwas(pop.geno, pop.pheno, marker_map=mmap, n_pc=2)
    assert "chrom" in res.columns and "pos" in res.columns
    ax = gs.manhattan(res)
    assert ax is not None
    ax2 = gs.qq_plot(res)
    assert ax2 is not None


def test_gwas_handles_missing_phenotype():
    pop = gs.simulate_population(n=100, m=150, n_qtl=5, h2=0.5, seed=2)
    y = pop.pheno.copy(); y[:10] = np.nan
    res = gs.gwas(pop.geno, y)
    assert len(res) == 150
    assert np.isfinite(res["log10p"]).all()
