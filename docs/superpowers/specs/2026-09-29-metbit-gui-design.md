# metbit GUI - Design Spec

Date: 2026-09-29
Status: Approved design, pending implementation plan

## Goal

A desktop app with a modern, minimal GUI that runs the full metbit pipeline:
raw Bruker FID or CSV import, preprocessing, normalization, alignment, and
analysis (PCA, OPLS-DA, STOCSY, univariate stats). It ships as a single
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

## Screens

Layout: left pipeline rail, center plot, right parameter panel.

Rail status per stage: empty dot (not run), filled (done), pulsing (running),
red (failed), hollow ring (stale: an upstream stage was re-run).

| Stage | Center view | Parameters |
|---|---|---|
| Import | Sample table (name, group, path) | Bruker folder or CSV matrix; optional metadata CSV for groups |
| Process | Overlaid spectra, sample picker | ppm range, water region, bin size, phasing mode, baseline lambda |
| Normalize | Before / after spectra | PQN, MSC, TSP-area |
| Align | Before / after spectra (zoomable) | icoshift intervals, max shift |
| PCA | Scores / Loadings tabs | n components, scaling, color-by column |
| OPLS-DA | Scores / S-plot / VIP tabs | group pair, scaling, CV folds, VIP threshold |
| STOCSY | Correlation spectrum | driver ppm (or click on plot) |
| Stats | Volcano plot + results table | test type, FDR method |

CSV import skips Process and starts at Normalize.

`ParamPanel` renders its form from the JSON Schema that the server returns
for each stage (derived from Pydantic models). No hand-written per-stage forms.

Each stage has a `...` menu: export output matrix (CSV), plot (PNG/SVG),
fitted model (JSON).

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
| GET | `/stages/{id}/export?fmt=csv\|png\|svg\|json&path=` | Write export to a user-chosen path |
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
Parquet file per completed stage output. Loading restores the rail without
recomputing.

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

Each stage is independently shippable.

1. **Shell and bridge**: Electron, sidecar spawn and health check, rail,
   theme, CSV import, PCA. Proves the full path end to end.
2. **Analysis**: OPLS-DA, STOCSY, Stats, exports, project save/load, autosave.
3. **Preprocessing**: Bruker import and Process stage with per-sample progress.
4. **Normalize and Align**.
5. **Packaging CI**: GitHub Actions matrix (macos-14, windows-latest,
   ubuntu-latest). PyInstaller builds the sidecar, electron-builder bundles it
   into `resources/`.

## Out of scope (deferred)

- Code signing and notarization
- Auto-update
- Multiple concurrent sessions
- Remote or GPU job offload
