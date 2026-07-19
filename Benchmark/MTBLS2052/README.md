# MTBLS2052 validation — metbit raw-FID processing (5/6-Nx rat CKD)

`python Benchmark/MTBLS2052/validate_mtbls2052.py`
(archive in `Benchmark/data/MTBLS2052/`).

This dataset exercises the half ST002087 could not: **raw Bruker FID -> spectra**.
Original study: 5/6-nephrectomy rat chronic-kidney-disease model, multi-matrix
(urine, serum, kidney, lung, heart, spleen, liver), longitudinal weeks 0-6,
nephrectomy vs sham control (MetaboLights MTBLS2052).

## Archive verification

Downloaded `5_6_Nx_rat_CKD_model_archive.zip` (2.84 GB, byte-exact vs EBI FTP).
It is a zip-of-zips (one Bruker acquisition per sample):

- **750** valid 1D-FID sample zips (`<id>/fid`, CPMG / NOESY / qNOESY)
- **326** valid 2D JRES zips (`<id>/ser`) — not used by metbit's 1D pipeline
- **4** genuinely corrupt entries (CRC-32 error in the source archive, e.g.
  `10030303.zip`); skipped automatically.

## Validation subset: kidney, week 6 (target organ)

34 samples (14 CKD / 20 sham) processed from raw FID with
`nmr_preprocessing(calib_type="tsp", auto_phasing=True, align=True)`.

### 1. Processing correctness
- Median TSP reference peak after calibration: **0.000 ppm** (within tolerance).
- Spectra chemically valid: sharp TSP singlet at 0, dense tissue-metabolite
  region 1-4.5 ppm, sparse aromatics, upright peaks, flat baseline
  (`result/spectra_qc_kidney_wk6.png`).

### 2. Biological separation (CKD vs sham)
- OPLS-DA: **R2Y = 0.970, Q2 = 0.697** — strong, cross-validated separation in
  the disease target organ, consistent with the study's reported renal
  metabolic phenotype.
- Top-VIP resonance clusters at ~2.25 ppm (plus 7.72 ppm).

## Verdict

metbit's raw-FID processing runs end-to-end on an independent Bruker dataset,
calibrates the reference to 0 ppm, and yields spectra that support a valid
CKD-vs-control model. Together with the ST002087 analysis-half validation, this
covers metbit from raw FID through chemometrics against independent data.

Note: MTBLS2052's MAF (metabolite result matrix) is empty on MetaboLights, so
metabolite-level concentration comparison is not possible here (that quantitative
check is done on ST002087); this dataset validates processing + separation.

## Outputs (`result/`)
- `spectra_qc_kidney_wk6.png`, `validation_report.json`
