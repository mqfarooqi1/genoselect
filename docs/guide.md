# User guide

## Installation

```bash
pip install genoselect
```

From source:

```bash
git clone https://github.com/mqfarooqi1/genoselect
cd genoselect && pip install -e ".[test]"
```

Requirements: Python 3.10+, NumPy, SciPy, scikit-learn, pandas.

## Data conventions

- **Genotypes** are a 2-D array of shape `(n_individuals, n_markers)`, coded as
  allele dosage `0`, `1`, or `2`.
- **Phenotypes** are a 1-D array of length `n_individuals`.
- Missing genotypes are `NaN`; impute them with `impute_markers`.

## Reading your data

All readers return a `GenotypeData` with `.geno` (n_samples × n_markers),
`.samples`, and `.markers`, and add no dependencies.

```python
import genoselect as gs

gd = gs.read_vcf("genotypes.vcf.gz")     # number of ALT alleles
gd = gs.read_plink("dataset")            # reads dataset.bed/.bim/.fam
gd = gs.read_hapmap("data.hmp.txt")      # HapMap text format

fit = gs.GBLUP().fit(gd.geno, phenotypes)
```

## GBLUP

```python
fit = gs.GBLUP(min_maf=0.0).fit(geno, y)
fit.predict(geno_new)
```

Fitted by REML (Endelman 2011 spectral method). Useful attributes:

| Attribute | Meaning |
|---|---|
| `h2_` | Estimated genomic heritability |
| `Vu_`, `Ve_`, `lambda_` | Genetic / residual variance and ratio |
| `gebv_` | Genomic estimated breeding values (training set) |
| `marker_effects_` | RR-BLUP marker effects (for prediction) |

A functional interface mirroring the R package is also available:

```python
fit = gs.gblup(y, geno=geno)   # fitted GBLUP
sol = gs.gblup(y, K=G)         # REML solution dict
```

## Models and cross-validation

```python
gs.available_models()
# ['gblup', 'elastic_net', 'random_forest', 'gradient_boosting', 'ensemble']

cv = gs.cross_validate(geno, y, models=["gblup", "elastic_net"],
                       k=5, scheme="kfold", random_state=0)
cv.summary()                   # model, mean, sd, n_folds
```

Accuracy is **predictive ability** — correlation between predicted and observed
phenotypes on held-out folds. Use `scheme="leave_group_out"` with a `groups`
array for forward-prediction scenarios. `benchmark(...)` runs all models.

## Stacked ensemble

```python
ens = gs.StackedEnsemble(inner_k=5, random_state=0).fit(geno, y)
ens.weights_                   # non-negative, sum to 1
ens.predict(geno_new)
```

A super-learner combining the base models with NNLS weights from an inner
cross-validation.

## GWAS

Single-marker association mapping with population-structure correction:

```python
res = gs.gwas(geno, y, marker_map=mmap, n_pc=3)   # mmap has chrom/pos columns
gs.manhattan(res)      # position-aware Manhattan plot
gs.qq_plot(res)        # calibration check
```

`gwas` returns a DataFrame with per-marker effect, standard error, t, p and
`log10p`. Plotting helpers require matplotlib (`pip install genoselect[plot]`).

## scikit-learn integration

```python
from sklearn.model_selection import cross_val_score
cross_val_score(gs.GBLUP(), geno, y, cv=5, scoring="r2")
```

## References

- VanRaden (2008), *J. Dairy Sci.* 91, 4414–4423. doi:10.3168/jds.2007-0980
- Endelman (2011), *Plant Genome* 4, 250–255. doi:10.3835/plantgenome2011.08.0024
- Meuwissen, Hayes & Goddard (2001), *Genetics* 157, 1819–1829.
- van der Laan, Polley & Hubbard (2007), *Stat. Appl. Genet. Mol. Biol.* 6(1).
