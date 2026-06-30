import numpy as np
import pytest
from sklearn.base import clone

import genoselect as gs


@pytest.fixture(scope="module")
def pop():
    return gs.simulate_population(n=150, m=400, n_qtl=30, h2=0.5, seed=0)


def test_simulate_shapes(pop):
    assert pop.geno.shape == (150, 400)
    assert pop.pheno.shape == (150,)
    assert set(np.unique(pop.geno)).issubset({0.0, 1.0, 2.0})


def test_grm_symmetric_unit_diagonal(pop):
    G = gs.vanraden_grm(pop.geno)
    assert G.shape == (150, 150)
    assert np.allclose(G, G.T)
    assert abs(np.mean(np.diag(G)) - 1.0) < 0.2


def test_gblup_fit_and_identity(pop):
    fit = gs.GBLUP().fit(pop.geno, pop.pheno)
    assert 0.0 <= fit.h2_ <= 1.0
    assert fit.gebv_.shape == (150,)
    W = pop.geno[:, fit._keep] - fit.marker_means_
    assert np.allclose(W @ fit.marker_effects_, fit.gebv_, atol=1e-8)
    assert np.corrcoef(fit.gebv_, pop.bv)[0, 1] > 0.4


def test_gblup_out_of_sample(pop):
    fit = gs.GBLUP().fit(pop.geno[:100], pop.pheno[:100])
    pred = fit.predict(pop.geno[100:])
    assert pred.shape == (50,)
    assert np.all(np.isfinite(pred))


def test_reml_orders_heritability():
    lo = gs.simulate_population(n=200, m=300, n_qtl=300, h2=0.2, seed=1)
    hi = gs.simulate_population(n=200, m=300, n_qtl=300, h2=0.8, seed=1)
    h2_lo = gs.GBLUP().fit(lo.geno, lo.pheno).h2_
    h2_hi = gs.GBLUP().fit(hi.geno, hi.pheno).h2_
    assert h2_hi > h2_lo


def test_available_and_make_model():
    names = gs.available_models()
    assert "gblup" in names and "ensemble" in names
    for n in names:
        gs.make_model(n)  # constructs without error
    with pytest.raises(ValueError):
        gs.make_model("not_a_model")


def test_cross_validate(pop):
    cv = gs.cross_validate(pop.geno, pop.pheno,
                           models=["gblup", "elastic_net"], k=5, random_state=0)
    s = cv.summary()
    assert set(s["model"]) == {"gblup", "elastic_net"}
    assert (s["n_folds"] == 5).all()
    assert np.isfinite(s["mean"]).all()


def test_ensemble_weights_sum_to_one(pop):
    ens = gs.StackedEnsemble(base_models=["gblup", "elastic_net"],
                             inner_k=3, random_state=0).fit(pop.geno[:100], pop.pheno[:100])
    assert abs(sum(ens.weights_.values()) - 1.0) < 1e-8
    pred = ens.predict(pop.geno[100:])
    assert pred.shape == (50,)


def test_qc_and_impute(pop):
    g = pop.geno.copy()
    g[0, 0] = np.nan
    out, keep = gs.qc_markers(g, maf=0.05, max_missing=0.1)
    assert not np.isnan(out).any()
    assert int(keep.sum()) == out.shape[1]


def test_sklearn_clone():
    est = gs.GBLUP(min_maf=0.01)
    cloned = clone(est)
    assert cloned.get_params()["min_maf"] == 0.01
