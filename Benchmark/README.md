# metbit Benchmark Suite

A multi-dataset benchmark instead of a single study, because each metbit
capability is best stressed by a different kind of data: raw FID processing,
alignment, PQN/MSC, PCA, OPLS-DA, VIP, and STOCSY. Each dataset targets one
tier; shared scoring lives in `metrics.py`.

## Tiers and datasets

Availability verified 2026-07 (see notes below each entry).

| Tier | Dataset | Accession / source | Exercises | Status |
|---|---|---|---|---|
| A. Correctness | NMRProcFlow Tomato | NMRProcFlow demo (`NMRFRIM3-4.zip` + `buckets_FRIM3-4.txt`) | Calibration → region exclusion → binning → normalisation → PCA, compared against the published bucket table | data is processed 1r, see A-note |
| B. QC / reproducibility | nPYc DEVSET | MetaboLights **MTBLS694** | Technical replicates, pooled QC (SR) + long-term reference (LTR): replicate CV, alignment, PQN/MSC, QC clustering, outlier detection | NMR-subset only, see B-note |
| C. Biological | CKD rat multi-matrix | MetaboLights **MTBLS2052** | Urine / serum / kidney / lung: per-matrix PCA, OPLS-DA, VIP stability, permutation, STOCSY | verified multi-matrix |
| D. Performance | COMETA COVID-19 plasma | Metabolomics Workbench **ST002087** (DOI 10.21228/M89T2Q) | FID throughput, RAM, four-tier backend scaling, PCA/OPLS-DA/VIP, cross-backend numerical agreement | verified raw FID |
| (CI) Regression | ChemoSpec metMUD1/2 | `ChemoSpec`/`SpecHelpers` R packages (simulated) | Unit/CI regression, backend numerical equivalence | CI-only (processed) |
| existing | MTBLS1 urine | MetaboLights **MTBLS1** | End-to-end pipeline + perf + reproducibility (already implemented in `MTBLS1/`) | done |

### Fully-public datasets (data AND reference results open) — verified 2026-07

For validating metbit against a reference, only these have BOTH the raw/processed
data and the reference results openly downloadable (no dead links, no paywalled
result tables):

| Dataset | Raw/processed data | Reference result | Both public? |
|---|---|---|---|
| **MTBLS12785** (urinary NMR birth cohort, atopy) | 196 per-sample Bruker **raw FID** zips | MAF: 39 identified metabolites + assigned ppm/multiplicity; open Nature Scientific Data descriptor | **YES** (best for the FID half) |
| **ST002087** (COMETA COVID plasma) | processed bucket tables (no FID) | mwTab: 139-metabolite µM matrix | **YES** (analysis half; already used) |
| MTBLS1 (urine) | raw FID | Salek 2007 results paywalled (MAF on EBI) | data yes, results partial |
| MTBLS2052 (rat CKD) | raw FID | Hanifa 2019 R2/Q2 **paywalled** | data yes, results NO |
| NMRProcFlow Tomato | Google Drive links **dead** | bucket table (also dead) | **NO** |
| breast-cancer MDPI (R2=0.846,Q2=0.770) | **not deposited** | reported in open paper | **NO** |

Recommendation: use **MTBLS12785** for a fully-public raw-FID validation (replaces
MTBLS2052, whose results are paywalled) and **ST002087** for the analysis half.
Together they cover metbit end-to-end with public data and public references.

### Verification notes (2026-07)

- **A / Tomato** — Solanum lycopersicum 'Moneymaker', 9 developmental stages x 2 technical replicates, Bruker Avance III 500.162 MHz. The demo ZIP holds **processed `1r` spectra, not raw FID**, and is hosted on Google Drive (`NMRFRIM3-4.zip`, `samples_p1.txt`, `buckets_FRIM3-4.txt`, `wb_NMRFRIM3-4.xlsx`) with no DOI/FTP. Use it for correctness of calibration, bucketing, normalisation and PCA against the reference bucket table — **not** for raw-FID processing (use MTBLS1 / ST002087 for that). Mirror the files locally since Drive links are unstable.
- **B / MTBLS694** — confirmed nPYc DEVSET: 6 pooled human-urine sources, each prepared and measured 13x, plus SR pooled QC and LTR. Cross-platform study, so its FTP `FILES/` is dominated by **~940 LC-MS `*.raw.zip` at 1.2-1.4 GB each (~1 TB, irrelevant to metbit)**. Download only the NMR Bruker subset listed in `a_MTBLS694_NMR_*.txt` (the "Raw Spectral Data File" column), not the whole study.
- **C / MTBLS2052** — verified matrices: urine (~340), blood serum (~90), kidney (~50), lung (~40); ~520 samples, weeks 0-6 longitudinal, experiments CPMG/JRES/NOESY/qNOESY. Raw is a single archive `5_6_Nx_rat_CKD_model_archive.zip` (2.6 GB). Analyse each matrix separately; note the repeated-measures (per-rat, multi-timepoint) design when doing cross-validation, same caveat as MTBLS1.
- **D / ST002087** — CORRECTION after downloading: despite the study page labelling raw data as "fid", `ST002087_Data.zip` (6.1 MB) contains **only processed bucket tables** (CPMG / NOESY / diffusion, 368 x 490 buckets), no Bruker FID. It DOES ship the article's named-metabolite result matrix (139 metabolites x 368 samples, µM) via mwTab. Downloaded to `data/ST002087/` (see its `PROVENANCE.md`). Use it to validate metbit's normalisation → PCA → OPLS-DA → VIP against the article results — **not** FID preprocessing. Raw-FID processing stays on MTBLS1 (local) / MTBLS2052.
- **CI / metMUD** — ships as already-processed `Spectra` objects in ChemoSpec, not Bruker FID; CI/regression and backend-equivalence only, not a preprocessing-correctness reference.

## Metrics (`metrics.py`)

Import from any per-dataset script: `from Benchmark.metrics import rmse, ...`

- **Correctness** — `rmse`, `reference_correlation`, `integrated_area_relative_error`,
  `negative_area_fraction`, `alignment_error_ppm`
- **QC** — `median_feature_cv` (before/after normalisation), `replicate_mean_cv`,
  `pairwise_replicate_correlation`, `qc_cluster_dispersion`
- **Chemometrics** — R2X/R2Y/Q2 (from `opls_da`), `classification_metrics`
  (balanced accuracy, sensitivity, specificity, ROC-AUC), `vip_rank_correlation`,
  `bootstrap_vip_stability`
- **Performance** — `throughput`, `speedup`, `max_abs_error`, plus wall-clock and
  `tracemalloc` peak memory (see `MTBLS1/benchmark_mtbls1.py: timed_stage`)

Preprocessing quality is also summarised by RMSE against a reference spectrum:

```
RMSE = sqrt( (1/n) * sum_i (x_i - x_ref_i)^2 )
```

## Layout

```
Benchmark/
  metrics.py            # shared scoring, self-checked (python Benchmark/metrics.py)
  README.md             # this file
  MTBLS1/               # existing end-to-end + performance + reproducibility tier
  <dataset>/            # one dir per dataset: benchmark_<name>.py + result/ + README.md
```

Each per-dataset directory follows the MTBLS1 template: a single runnable
`benchmark_<name>.py` that writes JSON reports + plots into `result/`, and a
`build_report.py` that renders a self-contained `report.html`.

## Suggested download / build order

1. **NMRProcFlow Tomato** — smallest; confirms import + preprocessing correctness against a known reference.
2. **MTBLS694 (nPYc DEVSET)** — QC and normalisation.
3. **ST002087 (COMETA)** — large-scale performance and backend scaling.
4. **MTBLS2052 (CKD)** — biological interpretation across matrices.

Raw data is not committed (large); each dataset README records the fetch step,
mirroring the MTBLS1 convention.
