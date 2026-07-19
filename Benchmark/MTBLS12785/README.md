# MTBLS12785 validation — metbit vs an open assignment reference

`python Benchmark/MTBLS12785/validate_mtbls12785.py`
(data in `Benchmark/data/MTBLS12785/`).

The fully-public dataset: **both** halves are open — raw Bruker FID (662
infant-urine acquisitions across ages 1M-60M) and a MAF listing 39 identified
metabolites with their assigned 1H ppm. Open-access Nature Scientific Data
descriptor; no paywall, no dead links. This is the clean replacement for the
paywalled-result MTBLS2052.

## Validation (1-month urine, n=60)

### Processing correctness
- Raw FID processed with metbit (`auto_phasing=True`, `calib_type="tsp"`, `align=True`).
- Median TSP reference peak: **-0.0004 ppm** (calibration correct).

### Peak-assignment concordance vs the article's MAF
- **16 / 39** assigned metabolite peaks resolved (SNR>3, local noise) in the
  mean spectrum. The resolved set is the abundant infant-urine metabolites
  (creatinine, citrate, dimethylamine, N,N-dimethylglycine, taurine, alanine,
  hippurate, pyruvate, ...).
- For those resolved peaks, metbit's peak position matches the article's
  assigned ppm to **median 0.0075 ppm (max 0.017 ppm)** — metbit reproduces the
  published assignments to within ~7 milli-ppm.
- Unresolved assignments are low-abundance / dietary / microbial metabolites
  (trigonelline, 1-methylnicotinamide, aromatic acids, dietary sugars). The MAF
  spans ages 0-5 years; these are expectedly weak or absent in 1-month milk-fed
  infant urine, so "unresolved" reflects biology, not a processing error.

## Verdict

On a dataset where the data and the reference are both public, metbit's
raw-FID processing calibrates correctly and places the abundant metabolites'
peaks on the article's assigned chemical shifts to within ~0.008 ppm. Combined
with ST002087 (open concentration matrix, analysis half) this gives a
fully-public, end-to-end validation with no paywalled or dead sources.

## Outputs (`result/`)
- `peak_assignment_concordance.csv` (per-metabolite SNR + ppm offset)
- `mean_spectrum_assignments.png` (mean spectrum, assigned ppm marked)
- `validation_report.json`
