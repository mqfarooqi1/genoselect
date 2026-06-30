# genoselect

**scikit-learn-compatible genomic prediction and selection from SNP marker data.**

Genomic selection is dominated by R packages (rrBLUP, BGLR, sommer). `genoselect`
brings a clean, Pythonic, scikit-learn-compatible toolkit to the same problem:
fit, cross-validate, and benchmark genomic-prediction models from 0/1/2 marker
matrices, with a properly implemented **GBLUP by REML** at its core — validated
against rrBLUP to ~1e-6.

## Install

```bash
pip install genoselect
```

## Quick start

```python
import genoselect as gs

pop = gs.simulate_population(n=250, m=800, n_qtl=40, h2=0.5, seed=1)

fit = gs.GBLUP().fit(pop.geno, pop.pheno)   # GBLUP by REML
print(fit.h2_)                              # genomic heritability
pred = fit.predict(pop.geno[:10])           # predict new genotypes

cv = gs.benchmark(pop.geno, pop.pheno, k=5, random_state=1)
print(cv.summary())                         # compare models
```

Load your own data:

```python
gd = gs.read_vcf("genotypes.vcf.gz")   # or read_plink(...), read_hapmap(...)
fit = gs.GBLUP().fit(gd.geno, phenotypes)
```

## Where to go next

- **[Tutorial](tutorial.ipynb)** — a full worked example with a model benchmark
  and the rrBLUP validation.
- **[User guide](guide.md)** — installation, data conventions, and every function.
- **[API reference](api.md)** — auto-generated from the source.
- **[PDF guide](genoselect-guide.pdf)** — the user guide as a printable PDF.

## Features

- VanRaden genomic relationship matrix
- **GBLUP by REML** (Endelman spectral method) + RR-BLUP marker effects
- A common interface to GBLUP, elastic net, random forest, gradient boosting,
  and a **stacked super-learner ensemble**
- Breeding-relevant cross-validation (k-fold, leave-group-out)
- One-line readers for **VCF / PLINK / HapMap**
- Population simulation and quality control

Released under the MIT License · [GitHub](https://github.com/mqfarooqi1/genoselect)
