# ST002087 validation — metbit vs the COMETA article

`python Benchmark/ST002087/validate_st002087.py` (data in `Benchmark/data/ST002087/`).

Because the deposit ships processed bucket tables (no FID) plus the article's
quantified metabolite matrix, this validates metbit's analysis half:
normalisation → PCA → OPLS-DA → VIP.

## 1. Feature fidelity (CPMG, n≈365 samples)

metbit's PQN-normalised bucket intensity at each metabolite's known 1H ppm vs
the article's independently quantified concentration:

| Metabolite | ppm | Pearson r | Spearman r |
|---|---|---|---|
| Glucose | 5.23 | 0.985 | 0.918 |
| Alanine | 1.48 | 0.915 | 0.908 |
| Pyruvate | 2.37 | 0.917 | 0.900 |
| 3-Hydroxybutyrate | 1.20 | 0.961 | 0.767 |
| Lactate | 1.33 | 0.753 | 0.723 |
| Creatinine | 3.04 | 0.234 | 0.214 |
| Citrate | 2.54 | 0.017 | 0.129 |
| Acetate | 1.92 | 0.169 | 0.106 |

Median |Spearman| = 0.745. metbit's spectral features strongly track the
article's concentrations for the abundant, well-resolved metabolites, and are
weak exactly where chemistry predicts (creatinine/citrate/acetate are low and
overlapped in plasma). NOESY gives lower agreement (median 0.214) as expected,
since CPMG suppresses the broad protein/lipid background over these peaks.

## 2. Discrimination (acute COVID-19<21d, n=246 vs Post-COVID-19, n=94)

| Experiment | R2Y | Q2 |
|---|---|---|
| CPMG | 0.704 | 0.562 |
| NOESY | 0.700 | 0.608 |

Q2 > 0.5 indicates genuine, cross-validated separation. On CPMG the top-VIP
regions cluster at 0.6-0.9 ppm (lipoprotein/lipid CH3 and CH2 envelope), plus
glucose (5.27) and 1.23 ppm — consistent with the article's emphasis on
lipoprotein remodelling as the dominant COVID-19 plasma signature.

## Verdict

metbit reproduces the article's quantitative structure: its processed features
correlate with the published concentrations (up to r=0.99 for abundant
metabolites) and its supervised model recovers a valid acute-vs-post separation
weighted on the same lipoprotein region the article highlights. This validates
metbit's normalisation and chemometrics against an independent reference. FID
preprocessing is not exercised here (no raw FID in the deposit); that is covered
by MTBLS1 and, once downloaded, MTBLS2052.

## Outputs (`result/`)
- `feature_fidelity_cpmg.csv`, `feature_fidelity_noesy.csv`
- `feature_fidelity_cpmg.png`
- `validation_report.json`
