# genoselect

**scikit-learn-compatible genomic prediction and selection from SNP marker data.**

[![PyPI](https://img.shields.io/pypi/v/genoselect)](https://pypi.org/project/genoselect/)
[![Python](https://img.shields.io/pypi/pyversions/genoselect)](https://pypi.org/project/genoselect/)
[![Downloads](https://img.shields.io/pypi/dm/genoselect)](https://pypi.org/project/genoselect/)
[![tests](https://github.com/mqfarooqi1/genoselect/actions/workflows/tests.yml/badge.svg)](https://github.com/mqfarooqi1/genoselect/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21221653.svg)](https://doi.org/10.5281/zenodo.21221653)

Genomic selection is dominated by R packages (rrBLUP, BGLR, sommer). `genoselect`
brings a clean, Pythonic, `scikit-learn`-compatible toolkit to the same problem:
fit, cross-validate, and benchmark genomic-prediction models from 0/1/2 marker
matrices, with a properly implemented **GBLUP by REML** at its core.

## Features

- **VanRaden genomic relationship matrix** [`vanraden_grm`].
- **GBLUP by REML** — the Endelman (2011) spectral method, with equivalent
  ridge-regression marker effects for out-of-sample prediction (`GBLUP`).
- **A common model interface** — GBLUP, elastic net, random forest, gradient
  boosting, and a **stacked super-learner ensemble**, all as scikit-learn
  estimators (`make_model`, `available_models`).
- **Breeding-relevant cross-validation** — k-fold and leave-group-out, reporting
  predictive ability honestly (`cross_validate`, `benchmark`).
- **GWAS** — single-marker association (`gwas`) with PC structure correction,
  plus position-aware `manhattan` and `qq_plot` helpers.
- **Read your own data** — `read_vcf`, `read_plink`, `read_hapmap` load VCF /
  PLINK / HapMap files into 0/1/2 matrices (pure-Python, no extra deps).
- **Simulation and QC** — `simulate_population`, `qc_markers`, `impute_markers`.

## Install

```bash
pip install genoselect
```

Or from source:

```bash
git clone https://github.com/mqfarooqi1/genoselect
cd genoselect && pip install -e .
```

## Quick start

```python
import genoselect as gs

# simulate a population with a heritable trait
pop = gs.simulate_population(n=250, m=800, n_qtl=40, h2=0.5, seed=1)

# fit GBLUP and inspect heritability + breeding values
fit = gs.GBLUP().fit(pop.geno, pop.pheno)
print(fit.h2_)            # estimated genomic heritability
print(fit.gebv_[:5])      # genomic estimated breeding values

# predict new genotypes
pred = fit.predict(pop.geno[:10])

# benchmark several models by cross-validation
cv = gs.benchmark(pop.geno, pop.pheno, k=5, random_state=1)
print(cv.summary())
```

Load your own data instead of simulating:

```python
gd = gs.read_vcf("genotypes.vcf.gz")     # or read_plink("data"), read_hapmap("data.hmp.txt")
fit = gs.GBLUP().fit(gd.geno, phenotypes)
```

Because the estimators follow the scikit-learn API, they compose with the wider
ecosystem (`Pipeline`, `GridSearchCV`, `cross_val_score`, …).

## Methods

The genomic relationship matrix follows VanRaden (2008,
[doi:10.3168/jds.2007-0980](https://doi.org/10.3168/jds.2007-0980)); the
mixed-model solver follows Endelman (2011,
[doi:10.3835/plantgenome2011.08.0024](https://doi.org/10.3835/plantgenome2011.08.0024));
the genomic-selection framework follows Meuwissen, Hayes & Goddard (2001,
[doi:10.1093/genetics/157.4.1819](https://doi.org/10.1093/genetics/157.4.1819)).

## Tests

```bash
pip install -e ".[test]"
pytest
```

---

Author: **Muhammad Farooqi** · MIT licensed. A Python companion to the R package
[GSbench](https://github.com/mqfarooqi1/GSbench).
