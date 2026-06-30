"""Readers for common genomic file formats into 0/1/2 dosage matrices.

All readers return a :class:`GenotypeData` with ``geno`` of shape
``(n_samples, n_markers)`` coded as allele dosage (0/1/2; missing as NaN),
plus the sample and marker identifiers. They are pure-Python/NumPy and add no
dependencies; for very large files a specialised reader (scikit-allel,
bed-reader) may be faster.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass

import numpy as np

__all__ = ["GenotypeData", "read_vcf", "read_hapmap", "read_plink"]


@dataclass
class GenotypeData:
    """Genotypes plus identifiers.

    ``geno`` is ``(n_samples, n_markers)`` allele dosage (0/1/2, NaN = missing).
    """
    geno: np.ndarray
    samples: list
    markers: list

    def __repr__(self) -> str:
        n, m = self.geno.shape
        return f"<GenotypeData: {n} samples x {m} markers>"


def _open(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def read_vcf(path) -> GenotypeData:
    """Read a (optionally gzipped) VCF into dosages = number of ALT alleles.

    Biallelic diploid sites are used; multiallelic sites are skipped.
    """
    samples: list = []
    markers: list = []
    rows: list = []
    with _open(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            if not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            chrom, pos, vid, _ref, alt = f[0], f[1], f[2], f[3], f[4]
            if "," in alt:                       # skip multiallelic
                continue
            markers.append(vid if vid not in (".", "") else f"{chrom}:{pos}")
            fmt = f[8].split(":")
            gt_i = fmt.index("GT") if "GT" in fmt else 0
            dosages = []
            for cell in f[9:]:
                gt = cell.split(":")[gt_i].replace("|", "/")
                a = gt.split("/")
                try:
                    if a[0] in (".", "") or (len(a) > 1 and a[1] == "."):
                        dosages.append(np.nan)
                    elif len(a) > 1:
                        dosages.append(float(int(a[0]) + int(a[1])))
                    else:
                        dosages.append(float(2 * int(a[0])))
                except ValueError:
                    dosages.append(np.nan)
            rows.append(dosages)
    geno = np.array(rows, dtype=float).T if rows else np.empty((len(samples), 0))
    return GenotypeData(geno=geno, samples=samples, markers=markers)


def read_hapmap(path) -> GenotypeData:
    """Read a HapMap file into dosages = copies of the second listed allele."""
    samples: list = []
    markers: list = []
    rows: list = []
    with _open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        samples = header[11:]
        for line in fh:
            if not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            markers.append(f[0])
            alleles = f[1].split("/")
            alt = alleles[1] if len(alleles) > 1 else None
            dosages = []
            for g in f[11:]:
                g = g.strip().upper()
                if alt is None or len(g) < 2 or "N" in g:
                    dosages.append(np.nan)
                else:
                    dosages.append(float(sum(ch == alt for ch in g[:2])))
            rows.append(dosages)
    geno = np.array(rows, dtype=float).T if rows else np.empty((len(samples), 0))
    return GenotypeData(geno=geno, samples=samples, markers=markers)


def read_plink(prefix) -> GenotypeData:
    """Read a PLINK 1 binary fileset (``.bed``/``.bim``/``.fam``).

    Dosage is the number of copies of the ``.bim`` A1 allele (0/1/2, NaN
    missing). Only SNP-major ``.bed`` files are supported (the PLINK default).
    """
    prefix = str(prefix)
    markers: list = []
    with open(prefix + ".bim") as fh:
        for line in fh:
            cols = line.split()
            if cols:
                markers.append(cols[1])
    samples: list = []
    with open(prefix + ".fam") as fh:
        for line in fh:
            cols = line.split()
            if cols:
                samples.append(cols[1])

    n, m = len(samples), len(markers)
    with open(prefix + ".bed", "rb") as fh:
        magic = fh.read(3)
        if magic[:2] != b"\x6c\x1b":
            raise ValueError("Not a PLINK .bed file (bad magic bytes).")
        if magic[2] != 1:
            raise ValueError("Only SNP-major .bed files are supported.")
        raw = np.frombuffer(fh.read(), dtype=np.uint8)

    bytes_per_snp = (n + 3) // 4
    raw = raw.reshape(m, bytes_per_snp)
    codes = np.zeros((m, bytes_per_snp * 4), dtype=np.uint8)
    for k in range(4):
        codes[:, k::4] = (raw >> (2 * k)) & 0b11
    codes = codes[:, :n]

    dosage = np.full(codes.shape, np.nan)
    dosage[codes == 0] = 2.0   # homozygous A1
    dosage[codes == 2] = 1.0   # heterozygous
    dosage[codes == 3] = 0.0   # homozygous A2
    # codes == 1 -> missing (NaN)
    return GenotypeData(geno=dosage.T, samples=samples, markers=markers)
