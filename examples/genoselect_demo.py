"""
genoselect — comprehensive demonstration script
================================================

Exercises every public function of genoselect on a small DUMMY SNP dataset:
    simulate_population, qc_markers, impute_markers, centre_markers,
    vanraden_grm, GBLUP (+ predict, attributes), gblup, reml_solve,
    available_models, make_model, cross_validate, benchmark, CVResult.summary,
    StackedEnsemble (+ predict), read_vcf / read_plink / read_hapmap.

Run it with:
    python examples/genoselect_demo.py
(after `pip install genoselect`)
"""
import tempfile
from pathlib import Path

import numpy as np

import genoselect as gs

rng = np.random.default_rng(1)
print("genoselect version:", gs.__version__)

# ---------------------------------------------------------------------------
# 1. Create a dummy SNP dataset
#    rows = individuals, columns = SNP markers coded as allele dosage 0/1/2.
# ---------------------------------------------------------------------------
n_ind, n_marker = 150, 400
freqs = rng.uniform(0.05, 0.95, size=n_marker)
geno = rng.binomial(2, freqs, size=(n_ind, n_marker)).astype(float)

# a trait controlled by 30 causal markers + noise (h2 ~= 0.5)
qtl = rng.choice(n_marker, size=30, replace=False)
gv = (geno[:, qtl] - 2 * freqs[qtl]) @ rng.normal(size=30)
gv = gv / gv.std()
ve = (1 - 0.5) / 0.5
pheno = gv + rng.normal(scale=np.sqrt(ve), size=n_ind)

# sprinkle in missing genotypes for the QC demo
miss = (rng.integers(0, n_ind, 60), rng.integers(0, n_marker, 60))
geno[miss] = np.nan
print(f"Dummy data: {n_ind} individuals x {n_marker} markers; "
      f"{int(np.isnan(geno).sum())} missing genotypes")

# ---------------------------------------------------------------------------
# 2. Quality control & imputation
# ---------------------------------------------------------------------------
X, keep = gs.qc_markers(geno, maf=0.05, max_missing=0.10, impute=True)
print(f"After QC: {X.shape[1]} markers retained; missing = {int(np.isnan(X).sum())}")

geno_imp = gs.impute_markers(geno)   # impute without filtering
print("After imputation only, missing =", int(np.isnan(geno_imp).sum()))

# ---------------------------------------------------------------------------
# 3. Relationship matrix (VanRaden) and marker centring
# ---------------------------------------------------------------------------
G = gs.vanraden_grm(X)
print(f"GRM: {G.shape}, mean diagonal = {np.mean(np.diag(G)):.3f}")

W, p, c, kept = gs.centre_markers(X)     # centred markers + VanRaden scaling
print(f"centred markers: {W.shape}, normaliser c = {c:.1f}")

# ---------------------------------------------------------------------------
# 4. GBLUP by REML (scikit-learn estimator) + out-of-sample prediction
# ---------------------------------------------------------------------------
test = rng.choice(n_ind, size=30, replace=False)
train = np.setdiff1d(np.arange(n_ind), test)

fit = gs.GBLUP().fit(X[train], pheno[train])
print(f"\nGBLUP: h2 = {fit.h2_:.3f}  (Vu = {fit.Vu_:.3f}, Ve = {fit.Ve_:.3f})")
print("first GEBVs:", np.round(fit.gebv_[:4], 3))
pred = fit.predict(X[test])
print("GBLUP test accuracy (corr):", round(float(np.corrcoef(pred, pheno[test])[0, 1]), 3))

# functional interface (mirrors R): geno form returns a fitted GBLUP,
# K form returns the raw REML solution dict
fit2 = gs.gblup(pheno[train], geno=X[train])
sol = gs.gblup(pheno[train], K=gs.vanraden_grm(X[train]))
print("reml_solve h2:", round(sol["Vu"] / (sol["Vu"] + sol["Ve"]), 3))

# reml_solve directly on a relationship matrix
sol2 = gs.reml_solve(pheno[train], gs.vanraden_grm(X[train]))
print("direct reml_solve lambda:", round(sol2["lam"], 3))

# ---------------------------------------------------------------------------
# 5. Model registry
# ---------------------------------------------------------------------------
print("\navailable models:", gs.available_models())
enet = gs.make_model("elastic_net", random_state=1)
enet.fit(X[train], pheno[train])
print("elastic-net test accuracy:",
      round(float(np.corrcoef(enet.predict(X[test]), pheno[test])[0, 1]), 3))

# ---------------------------------------------------------------------------
# 6. Cross-validation across models
# ---------------------------------------------------------------------------
cv = gs.cross_validate(
    X, pheno,
    models=["gblup", "elastic_net", "random_forest", "gradient_boosting"],
    k=5, random_state=1,
)
print("\nk-fold cross-validation:")
print(cv.summary().to_string(index=False))

# leave-group-out (e.g. families / environments)
groups = np.tile(np.arange(5), n_ind // 5 + 1)[:n_ind]
cv_lgo = gs.cross_validate(X, pheno, models=["gblup", "elastic_net"],
                           scheme="leave_group_out", groups=groups)
print("\nleave-group-out cross-validation:")
print(cv_lgo.summary().to_string(index=False))

# benchmark() = all models at once
bench = gs.benchmark(X, pheno, k=5, random_state=1)
print("\nbenchmark (all models):")
print(bench.summary().to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Stacked super-learner ensemble
# ---------------------------------------------------------------------------
ens = gs.StackedEnsemble(random_state=1).fit(X[train], pheno[train])
print("\nensemble weights:", {k: round(v, 3) for k, v in ens.weights_.items()})
print("ensemble test accuracy:",
      round(float(np.corrcoef(ens.predict(X[test]), pheno[test])[0, 1]), 3))

# ---------------------------------------------------------------------------
# 8. Built-in simulator
# ---------------------------------------------------------------------------
pop = gs.simulate_population(n=120, m=500, n_qtl=50, h2=0.5, seed=1)
print("\nsimulated population:", pop.geno.shape,
      "| GBLUP h2 =", round(gs.GBLUP().fit(pop.geno, pop.pheno).h2_, 3))

# ---------------------------------------------------------------------------
# 9. Reading real file formats (dummy VCF shown; PLINK/HapMap identical)
# ---------------------------------------------------------------------------
vcf_text = (
    "##fileformat=VCFv4.2\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\tC\n"
    "1\t10\trs1\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1\n"
    "1\t20\trs2\tC\tT\t.\t.\t.\tGT\t1/1\t0/0\t0/1\n"
)
with tempfile.TemporaryDirectory() as d:
    vcf_path = Path(d) / "dummy.vcf"
    vcf_path.write_text(vcf_text)
    gd = gs.read_vcf(vcf_path)
    print("\nread_vcf ->", gd, "\nsamples:", gd.samples, "markers:", gd.markers)
    print("dosages:\n", gd.geno)
# gs.read_plink("prefix")   -> reads prefix.bed/.bim/.fam
# gs.read_hapmap("x.hmp.txt")

print("\n==== genoselect demonstration complete ====")
