# metbit MTBLS1 Full-Pipeline Benchmark

Reproduces the full metbit workflow on the public MTBLS1 human urine <sup>1</sup>H NMR
study (Salek et al., 2007; via MetaboLights), from raw Bruker FID archives
through preprocessing, alignment, normalisation, PCA, OPLS-DA, VIP scoring,
and STOCSY. Referenced from the manuscript's Data Availability statement.

## Scope

- **Performance** — wall-clock time and peak memory (via `tracemalloc`) for
  each pipeline stage.
- **Reproducibility** — same-seed determinism check: the stochastic stages
  (OPLS-DA cross-validation) are re-fit `n=3` times against the same
  preprocessed matrix with a fixed `random_state`, comparing R2Y, Q2, and the
  top-10 VIP-ranked ppm positions across runs.
- **Usability** — the pipeline is a single Python function; the report
  records its length and the number of distinct metbit API calls needed to
  go from raw FID directories to a STOCSY connectivity plot, with no manual
  intervention between stages.

## Running

```bash
python Benchmark/MTBLS1/benchmark_mtbls1.py
```

Requires the raw MTBLS1 FID archive (132 Bruker directories) at
`manuscript/benchmark/data/MTBLS1/FILES` and sample metadata at
`manuscript/benchmark/data/MTBLS1/samples_parsed.csv`. These are fetched from
MetaboLights separately (not part of this script, to avoid re-downloading
~132 FID directories on every run) and are gitignored due to size.

## Output (`result/`)

| File | Contents |
|---|---|
| `REPORT.md` | Combined human-readable summary of all three scopes |
| `performance_report.json` | Per-stage timing, peak memory, environment info |
| `reproducibility_report.json` | Same-seed determinism metrics |
| `usability_report.json` | API-call / lines-of-code usability metrics |
| `opls_metrics.json` | R2Y, Q2, groups, top-VIP STOCSY anchor |
| `spectral_matrix_normalised.csv` | PQN-normalised, aligned spectral matrix (float32) |
| `pca_scores.csv` | PCA scores (PC1-PC5) per sample |
| `vip_scores.csv` | VIP score per spectral feature |

## Caveats

Disease status in MTBLS1 is completely confounded with acquisition batch
(see manuscript Section 2.13), and cross-validation here is performed at the
spectrum level, not the participant level. R2Y/Q2/VIP values in this
benchmark are workflow-execution outputs, not evidence of disease
discrimination — consistent with how they are reported in the manuscript.
