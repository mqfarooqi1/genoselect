"""Quick numerical validation of genoselect on simulated data."""
import numpy as np

import genoselect as gs

print("genoselect", gs.__version__)

# 1. simulate
pop = gs.simulate_population(n=250, m=800, n_qtl=40, h2=0.5, seed=1)
print("geno", pop.geno.shape, "| pheno", pop.pheno.shape)

# 2. GRM
G = gs.vanraden_grm(pop.geno)
print("GRM", G.shape, "mean diag %.3f" % np.mean(np.diag(G)))

# 3. GBLUP: heritability recovery + GEBV accuracy
fit = gs.GBLUP().fit(pop.geno, pop.pheno)
print("GBLUP h2 = %.3f (true 0.5)" % fit.h2_)
acc_train = np.corrcoef(fit.gebv_, pop.bv)[0, 1]
print("GEBV vs true BV corr = %.3f" % acc_train)

# identity check: W @ marker_effects == gebv
W = pop.geno[:, fit._keep] - fit.marker_means_
print("max|W@a - gebv| = %.2e" % np.max(np.abs(W @ fit.marker_effects_ - fit.gebv_)))

# 4. out-of-sample prediction (split the population)
tr, te = slice(0, 180), slice(180, 250)
fit2 = gs.GBLUP().fit(pop.geno[tr], pop.pheno[tr])
pred = fit2.predict(pop.geno[te])
print("out-of-sample pred vs true BV corr = %.3f" % np.corrcoef(pred, pop.bv[te])[0, 1])

# 5. cross-validation across models
cv = gs.cross_validate(pop.geno, pop.pheno,
                       models=["gblup", "elastic_net", "random_forest", "gradient_boosting"],
                       k=5, random_state=1)
print("\nCross-validation:")
print(cv.summary().to_string(index=False))

# 6. ensemble
ens = gs.StackedEnsemble(random_state=1).fit(pop.geno[tr], pop.pheno[tr])
ep = ens.predict(pop.geno[te])
print("\nensemble weights:", {k: round(v, 3) for k, v in ens.weights_.items()})
print("ensemble pred vs true BV corr = %.3f" % np.corrcoef(ep, pop.bv[te])[0, 1])

print("\nVALIDATION OK")
