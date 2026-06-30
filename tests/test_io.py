import numpy as np

import genoselect as gs


def test_read_vcf(tmp_path):
    vcf = tmp_path / "x.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
        "1\t100\trs1\tA\tG\t.\t.\t.\tGT\t0/0\t1/1\n"
        "1\t200\trs2\tC\tT\t.\t.\t.\tGT\t0/1\t./.\n"
        "1\t300\trs3\tA\tG\t.\t.\t.\tGT\t1|1\t0|1\n"
    )
    gd = gs.read_vcf(vcf)
    assert gd.samples == ["S1", "S2"]
    assert gd.markers == ["rs1", "rs2", "rs3"]
    expected = np.array([[0, 1, 2], [2, np.nan, 1]], dtype=float)
    np.testing.assert_array_equal(gd.geno, expected)


def test_read_hapmap(tmp_path):
    hmp = tmp_path / "x.hmp.txt"
    cols = "rs#\talleles\tchrom\tpos\tstrand\tassembly\tcenter\tprot\tassay\tpanel\tQC"
    hmp.write_text(
        cols + "\tS1\tS2\n"
        "m1\tA/G\t1\t100\t+\tNA\tNA\tNA\tNA\tNA\tNA\tAA\tGG\n"
        "m2\tC/T\t1\t200\t+\tNA\tNA\tNA\tNA\tNA\tNA\tCT\tNN\n"
    )
    gd = gs.read_hapmap(hmp)
    assert gd.samples == ["S1", "S2"]
    assert gd.markers == ["m1", "m2"]
    expected = np.array([[0, 1], [2, np.nan]], dtype=float)
    np.testing.assert_array_equal(gd.geno, expected)


def test_read_plink(tmp_path):
    p = tmp_path / "x"
    (tmp_path / "x.fam").write_text(
        "F1 S1 0 0 0 -9\nF1 S2 0 0 0 -9\nF1 S3 0 0 0 -9\n")
    (tmp_path / "x.bim").write_text(
        "1 snp1 0 100 A G\n1 snp2 0 200 C T\n")
    # snp1: S1=A1A1(2), S2=het(1), S3=A2A2(0) -> byte 0x38
    # snp2: S1=missing(NaN), S2=A1A1(2), S3=het(1) -> byte 0x21
    (tmp_path / "x.bed").write_bytes(bytes([0x6c, 0x1b, 0x01, 0x38, 0x21]))

    gd = gs.read_plink(str(p))
    assert gd.samples == ["S1", "S2", "S3"]
    assert gd.markers == ["snp1", "snp2"]
    expected = np.array([[2, np.nan], [1, 2], [0, 1]], dtype=float)
    np.testing.assert_array_equal(gd.geno, expected)


def test_reader_output_feeds_gblup(tmp_path):
    # a reader's output should plug straight into the estimators
    pop = gs.simulate_population(n=40, m=60, n_qtl=10, h2=0.5, seed=0)
    gd = gs.GenotypeData(geno=pop.geno, samples=[f"S{i}" for i in range(40)],
                         markers=[f"m{j}" for j in range(60)])
    fit = gs.GBLUP().fit(gd.geno, pop.pheno)
    assert fit.gebv_.shape == (40,)
