# metbit GUI - Design Spec

Date: 2026-09-29
Status: Approved. Stage 1 (shell, CSV import, PCA) implemented; full-pipeline revision approved 2026-09-29

## Goal

A desktop app with a modern, minimal GUI that runs the full metbit pipeline:
raw Bruker FID or CSV import, preprocessing, normalization, alignment,
multivariate analysis (PCA, OPLS-DA, PLS-R, LDA, ICA, HCA, STOCSY) and
statistics (volcano, ANOVA / Kruskal-Wallis). It ships as a single
installer with Python bundled, so users need no Python setup.

Targets: macOS arm64, Windows x64, Linux x64 (AppImage).

## Architecture

Electron shell + TypeScript/Preact renderer + frozen FastAPI sidecar.

```
GUI/
  electron/        # main process + preload (TypeScript)
  renderer/        # Vite + Preact UI (TypeScript, plain CSS)
    components/    # Rail, ParamPanel, PlotView, DataTable, JobBar
  server/          # FastAPI sidecar (Python, imports metbit)
    app.py         # routes
    session.py     # in-memory pipeline state
    jobs.py        # background job runner + progress events
  build/           # PyInstaller spec, electron-builder config
```

### Processes

- **Electron main**: picks a free localhost port, generates a random
  per-session token, spawns the `metbit-server` binary with both, polls
  `GET /health`, then opens the window. Port and token reach the renderer via
  `preload` (`contextBridge`). Kills the sidecar when the app quits.
- **Renderer**: `contextIsolation: true`, `nodeIntegration: false`. Talks to
  `http://127.0.0.1:<port>` with `Authorization: Bearer <token>`.
- **Sidecar**: binds `127.0.0.1` only. Holds one session. Each stage stores its
  output DataFrame in memory; downstream stages read it there. Matrices never
  cross into the renderer; only Plotly figure JSON and small tables do.

Plot logic is not reimplemented in JS: metbit's existing Plotly figures are
serialized with `fig.to_json()` and rendered with plotly.js.

## Pipeline model

Stages form a dependency graph in three rail sections:

```
DATA        Import -> Process (Bruker only) -> Normalize -> Align
ANALYSIS    PCA, OPLS-DA, PLS-R, LDA, ICA, HCA, STOCSY
STATISTICS  Volcano, Group tests (ANOVA / Kruskal-Wallis)
```

- **Working matrix**: every analysis reads the newest `done` data stage in the
  order Align, Normalize, Process, Import. Normalize and Align are optional.
- **Staleness**: each stage declares the stages it depends on (`after`).
  When a stage completes, every `done` stage that transitively depends on it
  becomes `stale`. Re-running an analysis never stales another analysis.
- **Import** has `format: csv | bruker`. CSV loads the spectral matrix (ppm
  column headers, other columns are metadata). Bruker lists experiment folders
  and joins a metadata CSV mapping sample to group; **Process** then runs
  `nmr_preprocessing` and is rejected for CSV imports.
- **Dynamic options**: fields tagged `options: "groups"` or
  `options: "columns"` get their `enum` filled by the server from the loaded
  data when `/stages` is listed.

## Screens

Layout: left pipeline rail (with section headers), center plot, right
parameter panel.

Rail status per stage: empty dot (not run), filled (done), pulsing (running),
red (failed), hollow accent ring (stale).

| Stage | Parameters | Views |
|---|---|---|
| Import | format, path (file or folder picker), metadata CSV, group column | Table |
| Process | bin size, auto-phasing, baseline type, calibration (TSP / DSS / custom) | Spectra, Table |
| Normalize | PQN, MSC, SNV, SNV+PQN | Before / after spectra |
| Align | ppm windows, reference (median / mean), max shift | Before / after spectra |
| PCA | components, scaling, PC axes | Scores, Loadings, Variance, Table |
| OPLS-DA | group A / B, scaling, CV folds, VIP threshold, permutations (0 = off) | Scores, S-plot, VIP, Permutation, Table |
| PLS-R | numeric target column, components, scaling | Predicted vs actual, Scores, Table |
| LDA | components, scaling | Scores, Loadings |
| ICA | components, scaling | Components, Mixing |
| HCA | linkage, metric, clusters | Dendrogram, Heatmap, Table |
| STOCSY | anchor ppm, p threshold | Correlation spectrum |
| Volcano | group A / B, fold-change and p thresholds, FDR method | Volcano, Table |
| Group tests | ANOVA or Kruskal-Wallis, FDR, bin width (default 0.01 ppm), boxplot feature | Ranked table, Boxplot |

- Clicking a point on a spectrum or loadings plot fills the current stage's
  ppm field when it has one (STOCSY anchor, Group tests feature).
- Group tests rank features on a binned matrix; the boxplot shows the chosen
  feature, or the top hit when none is given.
- OPLS-DA and Volcano compare two chosen groups; other samples are excluded.

`ParamPanel` renders its form from the JSON Schema that the server returns
for each stage (derived from Pydantic models). No hand-written per-stage forms.

Each stage has a `...` menu: export table or matrix (CSV), plot (PNG / SVG).

### Visual language

- Monochrome base, one accent: `#2563EB`.
- Inter, 13 px base, 8 px spacing grid.
- Hairline 1 px borders, no shadows, no card chrome.
- Light and dark themes via CSS custom properties, following
  `prefers-color-scheme`.

## API

All routes require the bearer token; missing or wrong token returns 401.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Readiness probe |
| GET | `/stages` | Stage list with param JSON schemas and status |
| POST | `/stages/{id}/run` | Validate params, start job, return `{job_id}` |
| GET | `/jobs/{id}/events` | SSE stream: `progress`, `done`, `error` |
| GET | `/stages/{id}/figure?view=<name>` | Plotly figure JSON |
| GET | `/stages/{id}/table?offset=&limit=` | Paginated summary table |
| POST | `/stages/{id}/export` | Write the stage table or matrix as CSV to a user-chosen `path` |
| POST | `/session/save` | Write `.metbit` project to a given path |
| POST | `/session/load` | Load `.metbit` project from a given path |

### Run flow

1. User clicks Run; renderer POSTs params.
2. Server validates with Pydantic (422 on failure), runs the metbit call on a
   worker thread. One session lock: only one job runs at a time; a second run
   request returns 409.
3. Renderer subscribes to SSE; `JobBar` shows percent (preprocessing reports
   per sample).
4. On `done`, output is stored, all downstream stages are marked stale, the
   renderer fetches the figure.

### Project file

`.metbit` is a zip: `params.json` (all stage params and statuses) plus one
Parquet file per completed data stage (Import, Process, Normalize, Align).
Fitted models are not stored: pickled objects would execute code when a
shared project is opened. On load, data stages restore directly and analysis
stages re-run from their saved params, except OPLS-DA permutation tests,
which load as `stale` until the user re-runs them.

Plot exports (PNG / SVG) are produced in the renderer with `Plotly.toImage`
and written by Electron after a native Save dialog; no image library is
needed in Python.

### File access

Only Electron opens native file dialogs. The server receives absolute paths
chosen by the user and only reads or writes those paths.

## Error handling

- Invalid params: 422, renderer highlights the offending fields.
- Job exception: SSE `error` event with short message and expandable
  traceback. Stage turns red; upstream outputs are kept.
- Sidecar crash: Electron detects process exit, shows a "Backend stopped"
  banner with Restart. The session autosaves to `userData/autosave.metbit`
  after each completed stage and is reloaded on restart.
- Startup: if `/health` is not ready within 20 s, show an error screen with
  the log path.
- Logs: `userData/logs/server.log`, rotated at 5 MB.

## Testing

- **Server**: pytest + FastAPI `TestClient` in `tests/gui/`. Per stage: run
  on a small synthetic matrix. Also: token rejection, 409 on concurrent run,
  stale propagation, save/load round trip. Runs in existing CI.
- **Renderer**: one Vitest file covering `ParamPanel` schema-to-form.
- **E2E**: one Playwright-Electron smoke test: launch, import CSV, run PCA,
  assert a plot renders. Runs per OS in the packaging workflow.

## Build order

Each phase is independently shippable and gets its own plan, written just
before it runs.

1. **Stage 1 (done)**: shell and bridge, CSV import, PCA.
2. **Phase A**: dependency-graph staleness, rail sections, dynamic options,
   Normalize, Align, OPLS-DA, STOCSY (click-to-pick), Volcano.
3. **Phase B**: PLS-R, LDA, ICA, HCA, Group tests.
4. **Phase C**: Bruker import and Process with per-sample progress, verified
   on the local MTBLS2052 subset (tests skip when absent).
5. **Phase D**: exports, project save / load, autosave and restore.
6. **Packaging CI**: GitHub Actions matrix (macos-14, windows-latest,
   ubuntu-latest). PyInstaller builds the sidecar, electron-builder bundles it
   into `resources/`. The CI must run `node node_modules/electron/install.js`
   if npm skips Electron's install script.

## Out of scope (deferred)

- Code signing and notarization
- Auto-update
- Multiple concurrent sessions
- Remote or GPU job offload
