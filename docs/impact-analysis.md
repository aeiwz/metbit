# Impact Analysis and Testing Strategy

## Component map
- `spec_norm.Normalization`: Normalisation utilities (PQN, SNV, MSC combinations) operating on pandas/NumPy inputs.
- `utility.lazypair`: Builds pairwise class splits and derived names for downstream OPLS/plotting flows.
- `utility.Normalise`: Higher-level normalisation helper with KNN imputation and multiple scaling options.
- `scaler.Scaler`: Custom sklearn-compatible scaler with configurable power scaling.
- Visualization/analysis modules (`metbit.metbit`, `lazy_opls_da`, plotting helpers): wrap sklearn/plotly; higher complexity and heavier data dependencies.

## Risk ranking
- High: `metbit.metbit`, `lazy_opls_da` (complex orchestration, external plotting, CV logic).
- Medium: `utility.Normalise`, `spec_norm.Normalization` (math correctness, NaN handling, scaling choices).
- Low: `utility.lazypair`, `scaler.Scaler` (deterministic transforms, limited surface).

## Change impact notes
- Scaling/normalisation changes ripple to downstream models: breaks expected ranges for OPLS/PCA visuals and VIP thresholds. Tests covering `Normalization`/`Normalise` mitigate this.
- `lazypair` column naming or pair ordering changes will alter file naming and subgroup analyses; tests assert validation and pair construction.
- Dependency bumps (sklearn, pandas, plotly) may alter defaults; watch for shape/dtype changes affecting scaler and imputer behaviours.
- Plotting/stat modules rely on numeric types; upstream type validation prevents silent coercion.

## Regression suite rules
- P0 (every PR): all unit tests in `tests/` (Normalization, lazypair, Normalise rounding/Z-score, Scaler). Target: keep runtime short and deterministic.
- P1 (daily/CI): PQN with imputation, edge-case scaling bounds, any future integration tests that touch file I/O or larger data fixtures.
- Coverage gate recommendation: 70%+ line coverage for core utilities; raise threshold as more modules gain tests.

## Gaps and next steps
- Add targeted tests for `metbit.metbit` (OPLS/PCA workflows) with small synthetic datasets to lock in metrics (R2/Q2) and plotting outputs.
- Cover `lazy_opls_da` path creation and HTML generation using tmp directories.
- Expand negative-path tests for plotting helpers (invalid markers, color maps) and error handling in CLI/UI wrappers.
- Introduce property-based tests for Normalisation functions to validate scale invariants across random inputs.
