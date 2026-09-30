# metbit GUI: project context

Handover notes for continuing work on the metbit desktop GUI after it moves out of the `metbit` repository into its own private repository (`metbit-gui`). Written 2026-09-30 at commit `8203a46` on branch `feat/gui` of the metbit repo.

## 1. What it is

A desktop app for NMR metabolomics built on the `metbit` Python library. It covers the whole pipeline: import (CSV table or raw Bruker FIDs), processing, normalization, alignment, multivariate models, statistics, peak annotation and time series. Everything runs locally.

- **Shell:** Electron 44 (main process in `electron/`).
- **Interface:** Preact 10 + TypeScript, built by Vite 8 (`renderer/`). Charts are Plotly.js 4.
- **Analysis:** a local Python "sidecar", FastAPI + uvicorn (`server/metbit_server/`), that calls the metbit library.
- **Targets:** macOS arm64, Windows, Linux. Bundling Python into installers is planned but **not done yet** (see section 8).
- **Credit:** "Developed by Kawa Technology". License MIT, copyright holder Theerayut (aeiwz) as in metbit's `LICENSE`.

## 2. Moving to the new repository: what breaks and how to fix it

The GUI folder is not self-contained yet. Fix these when you move it.

| # | Coupling today | File | What to do |
|---|---|---|---|
| 1 | The sidecar imports `metbit` from the metbit repo via `PYTHONPATH = <GUI>/../..` | `electron/main.ts` (`sidecarCommand`, dev branch) | Install metbit in the Python you run (`pip install -e /path/to/metbit` or a pinned version) and drop the `PYTHONPATH` line, or point it at an env var such as `METBIT_SRC`. `METBIT_PYTHON` already selects the interpreter. |
| 2 | The About page imports the repo root `LICENSE` (`../../../LICENSE?raw`) | `renderer/components/AboutPage.tsx` | Copy metbit's `LICENSE` to `GUI/LICENSE` (keep the text verbatim), change the import to `../../LICENSE?raw`. |
| 3 | Vite and Vitest allow reading one level above `GUI/` (for that LICENSE) | `vite.config.mts`, `vitest.config.mts` (`server.fs.allow`) | After step 2, remove the `server.fs.allow` lines. |
| 4 | The Python tests for the sidecar live in the metbit repo, not in `GUI/` | metbit `tests/gui/` (23 files incl. `conftest.py`) | Move them to `metbit-gui/tests/`. `conftest.py` puts `GUI/server` on `sys.path`; adjust that path. Add a `pytest.ini`; metbit's `pytest.ini` measured coverage of `metbit` only, so choose `--cov=metbit_server` if you want a gate here. |
| 5 | Done: design specs and implementation plans now live in `GUI/docs/` (`docs/specs/`, `docs/plans/`) and move with the folder | `docs/` | Inside the plans, file paths are written from the metbit repo root (`GUI/server/...`, `GUI/docs/...`); in the new repo read them without the `GUI/` prefix. |
| 6 | The GUI needs metbit library changes that exist only on metbit's `feat/gui` branch | metbit commit `b0272bc` (`metbit/analysis/stocsy.py`, `large_scale.py`): `STOCSY(..., plot="line" \| "markers", colorscale=...)`, gradient line plot | Merge metbit `feat/gui` (or at least `b0272bc`) and release metbit, then pin that version in the GUI. Without it the STOCSY stage fails (unknown keyword `plot`). |
| 7 | Ignore rules come from metbit's root `.gitignore` (`/GUI/node_modules/`, `/GUI/dist-electron/`, `/GUI/test-results/`, `dist/`, `build/`) | metbit `.gitignore` | Create `metbit-gui/.gitignore` with at least: `node_modules/`, `dist/`, `dist-electron/`, `test-results/`, `__pycache__/`, `.venv/`, and the heavy film outputs (item 8). |
| 8 | `metbit-film/` is inside `GUI/` and is **1.3 GB** (about 1.0 GB PNG frames, plus `web/node_modules`, `review/`, `output/*.mp4`, audio stems). It was never committed. | `metbit-film/` | Keep it out of git: ignore `metbit-film/frames/`, `metbit-film/review/`, `metbit-film/output/`, `metbit-film/web/node_modules/`, `metbit-film/web/dist/`, `metbit-film/audio/stems/`, or move the film to its own folder. The GUI only needs `assets/identity/` (already copied, about 60 KB). |
| 9 | Git history of the GUI is in the metbit repo (about 90 commits on `feat/gui`) | metbit repo | To keep history: `git subtree split --prefix=GUI -b gui-only` in the metbit repo, then pull that branch into the new repo. Note this carries only files under `GUI/`; items 4 to 6 are separate. |
| 10 | Helper script used while executing plans | metbit `.git/gui-tmp/extract.py` (writes files from plan code blocks, given a plan path from the repo root such as `GUI/docs/plans/...`), screenshots in `.git/gui-tmp/shots/` | Not needed at runtime. Copy if you keep writing plans in that style. |

Checks after the move (all must pass): `npm ci && npm run typecheck && npm test && npm run e2e`, and `python -m pytest tests` for the sidecar.

## 3. Architecture

```
Electron main (electron/main.ts)
  ├─ spawns the sidecar: python -m metbit_server --port <free> --autosave <userData>/autosave.metbit
  │     env METBIT_TOKEN=<random>; every HTTP call needs "Authorization: Bearer <token>"
  ├─ IPC: dialogs, file saves, project registry, app version, theme
  └─ BrowserWindow (contextIsolation, sandbox, no nodeIntegration) → dist/index.html
Preload (electron/preload.ts) exposes window.metbit.* (typed in renderer/types.d.ts)
Renderer (renderer/App.tsx) talks to the sidecar over http://127.0.0.1:<port> (renderer/api.ts)
Sidecar (server/metbit_server)
  ├─ app.py: FastAPI routes (below); Session with a stage dependency graph and staleness
  ├─ jobs.py: background jobs with SSE progress events
  ├─ stages/: one Stage per pipeline step (params = Pydantic model, run(), figures, table)
  ├─ annotations.py: peak labels, overlay on spectra, intensity table, annotator settings
  └─ project.py: .metbit project = zip of params.json + Parquet (never pickle)
```

**Stage model** (`stages/base.py`, `Stage` dataclass):
- Core fields: `id`, `title`, `section`, `params`, `run`.
- Data fields: `table`, `figures` (view name to figure function), `requires`, `after` (dependencies; re-running marks dependents stale).
- Click and view fields: `ppm_views` (views where a click yields a ppm), `pre_views` (views drawn before running), `companions` (main view to a clickable view drawn under it).
- Other fields: `blocked`, `export`.
- Param fields can carry schema hints the form understands: `options` (`groups`, `columns`, `time_columns`, `subject_columns`, `numeric_columns`), `pick`, `pick_ppm`, `show_if`, and `format` (`file-path`, `dir-path`).
- Figure functions may accept `style` and declared extras such as `ppm` or `mode`; `style.render()` forwards only what a function declares.

**Stages in rail order** (`stages/__init__.py`):
- **Data:** import, process (Bruker), normalize, align.
- **Analysis:** pca, opls, plsr, lda, ica, hca, stocsy.
- **Statistics:** volcano, grouptests, annotate.
- **Time series:** trajectory, timeprofile, asca, mixedmodel.

**Sidecar routes:**
- `/health`.
- Stages: `/stages`, `/stages/{id}/run`, `/jobs/{id}/events`.
- Stage outputs: `/stages/{id}/figure?view=&ppm=&mode=`, `/stages/{id}/html`, `/stages/{id}/table`, `/stages/{id}/export`.
- Style: `/style`, `/stages/{id}/style`.
- Annotations: `/annotations` (GET, POST), `/annotations/{id}` (PATCH, DELETE), `/annotations/import`, `/annotations/export`, `/annotations/settings` (GET, PUT).
- Session: `/session/save`, `/session/load`, `/session/restore`.

**Electron IPC:**
- Backend: `backend:info`, `backend:restart`.
- Files: `dialog:openFile`, `dialog:save`, `file:saveImage`.
- App: `app:version`, `theme:set`.
- Projects: `projects:list|add|remove|rename|duplicate|reveal|trash`. The registry is `electron/projects.ts`, stored in `<userData>/projects.json`. File actions only touch paths already in the list.

**Test seams** (env vars): `METBIT_E2E_DIR` (save dialogs return files here), `METBIT_E2E_OPEN` (open dialogs return this path), `METBIT_USER_DATA` (isolated userData), `METBIT_PYTHON` (interpreter).

## 4. Features, and where they live

- **Pages:** landing Home (new, open, continue, recent projects), Projects (search, open, rename, duplicate, show in folder, remove from list, move to Trash with confirmation), About & License (version, Kawa Technology credit, verbatim MIT text, open-source notices). Code: `renderer/components/{HomePage,ProjectsPage,AboutPage,PageNav}.tsx`. The About page table lists licenses checked against installed package metadata.
- **Import:** CSV or Bruker, with a file-format guide showing mock-ups and example data (`FormatGuide.tsx`, `exampleData.ts`).
- **Plot style:** global and per-stage overrides (`StyleForm.tsx`, `stages/style.py`). Group colors use the Okabe-Ito palette (colorblind safe).
- **STOCSY:** click the spectra to set the anchor, recent anchors, and a clear button (`StocsyTools.tsx`). The plot is the median spectrum as a line colored by r², with a colorbar (from the metbit library, item 6). Colour-scale choice is in the form.
- **Group tests:** boxplot with a clickable spectrum below it that picks the tested ppm. The tested bin is shaded, and your zoom is kept when you pick.
- **Annotate (metbit annotator dashboard UX):** toolbar (Single/Median, label font size, font family, Regular/Bold/Italic, current peak) and panels (add or edit a label, per group or sample line style, current labels, exports: intensity CSV, labels JSON, self-contained HTML). Labels are draggable (offset saved). Click a label on the plot or in the list to edit its text and angle.
- **Time series:** shared setup (time column plus order, optional subject column).
  - **Trajectory:** metbit `plot_pca_trajectory`, optional multilevel.
  - **Time profile.**
  - **ASCA:** effects, permutation p-values with within-subject shuffles, scores, loadings.
  - **Mixed model:** statsmodels MixedLM per binned feature, Wald test per term, FDR; Friedman option. Fits with `method=["bfgs","cg","nm"]`, because statsmodels' lbfgs fallback breaks on SciPy 1.18.
- **Projects and autosave:** the session autosaves after every successful job and is restored after a backend crash (Restart button).
- **Identity:** "resolved m" mark and lockup (`Brand.tsx`, geometry copied from the film's `generate_logo.py`), identity color tokens in `renderer/theme.css`, app icon `assets/icon.png`, favicon `renderer/favicon.svg`, sources in `assets/identity/`.
- **Theme:** System / Light / Dark switch (`ThemeToggle.tsx`, `themePref.ts`). Saved in localStorage, sets `data-theme` on `<html>` and Electron `nativeTheme`; plots redraw on change.
- **Decimal inputs:** every decimal field uses `DecimalInput.tsx`, which keeps exactly what is typed (so `0.05`, `-30`, `1e-7` can be entered).

## 5. Documents

- `context.md` (this file): handover and migration notes.
- `docs/specs/2026-09-29-metbit-gui-design.md`: the original GUI design (architecture, pipeline rail, stages A to D).
- `docs/specs/2026-09-29-metbit-gui-style-annotation-stocsy-design.md`: plot styling, annotation, STOCSY rework, group-test linking, annotator dashboard, time series (sections 1 to 8).
- `docs/plans/2026-09-29-metbit-gui-*.md`: step-by-step implementation plans, one per phase (stage 1, A to I).
- `metbit-film/`: the product film and the identity source (see section 2, item 8).

## 6. Commands

```bash
npm ci                      # install JS dependencies
npm start                   # build and launch (dev sidecar needs a Python with metbit + fastapi, uvicorn, pyarrow)
npm run typecheck           # tsc for renderer and electron
npm test                    # Vitest, 72 tests (renderer components + electron/projects)
npm run e2e                 # builds, then Playwright Electron tests, 15 specs (about 2.5 min)
SHOTS=<dir> npx playwright test e2e/layout.spec.ts   # overlap/clipping audit of every view at 1400x860 and 960x600, with screenshots
python -m pytest tests/gui --no-cov -q               # sidecar tests (144 at the time of writing; path changes after the move)
```

Running Playwright directly needs `npm run build` first, or it tests a stale bundle. Python deps for the sidecar: `server/requirements.txt` (metbit, fastapi, uvicorn, pyarrow). Developed with Node 26 and Python 3.14.

## 7. Conventions and decisions

- **Writing rules:** no em dash and no section-sign characters anywhere (code, comments, docs, commit messages). Commit messages carry no `Co-Authored-By` trailer. Keep files under 500 lines; `renderer/App.tsx` is about 660 lines and is the next thing to split.
- **Style of work:** minimal code ("ponytail"). Deliberate simplifications carry a `ponytail:` comment naming the limit and the upgrade path.
- **Tests:** test-first. Every feature gets unit tests plus an e2e check.
- **Layout audit:** the layout audit must report `{}` (no overlapping, clipped or wrapped content) at both window sizes.
- **Security:** bearer token on every sidecar route, sandboxed renderer, strict CSP in `renderer/index.html`, project files are zip + JSON + Parquet (no pickle), and labels are HTML-escaped and CSV formula-safe.
- **Accessibility:**
  - The identity blue `#3b82f6` is used for the logo only; it is 3.7:1 on white, below AA for text. Buttons use `#2563eb` (5.2:1). Links in dark mode use `#3b82f6` on navy (5.4:1).
  - Focus outlines on every control.
  - No native alert or confirm dialogs; inline confirmations instead.
- **Plots:** metbit figures are recolored from CSS tokens in `PlotView.tsx`. Paper-positioned notes are re-stacked in pixels (`lift_paper_annotations`) so they never collide. Exports (PNG/SVG) use the original white-background figure. `PlotView` keeps the user's zoom per `uirevision` itself, because Plotly's own `uirevision` drops the zoom on reversed ppm axes.
- **Landing:** the app opens on Home; the e2e specs click "New analysis" first.

## 8. Known limitations and open items

- **Packaging:** not done. No electron-builder config, bundled Python (PyInstaller sidecar) or CI release. `assets/icon.png` (512 px, transparent corners) is ready for it.
- **pingouin:** it is GPL-3.0, listed in metbit's `setup.py`, but imported nowhere. It will not be bundled (PyInstaller bundles imports only); consider removing it from metbit's requirements.
- **ASCA:** uses level means, which is exact for balanced designs and approximate otherwise (ASCA+ would use a GLM). The Group-effect permutation shuffles samples, not whole subjects.
- **Time order:** entered as comma-separated text, not a drag list.
- **Annotation fonts:** family and style apply to all labels, not per label.
- **Film README:** the title in `metbit-film/README.md` and `ART_DIRECTION.md` uses an em dash, and `metbit-film/identity/metbit-app-icon.png` has opaque corners. The GUI copy in `assets/` is fixed; the film originals are not.

## 9. History (metbit repo, branch `feat/gui`)

Built in phases, each with a spec section (`docs/specs/`) and a plan (`docs/plans/`):

| Phase | Scope |
|---|---|
| Stage 1, A to D | Shell, sidecar, stages, pipeline coverage, Bruker, exports, projects, autosave |
| E0 | Import format guide |
| E1 | Molecule loading animation |
| E, F, G | Plot style, peak annotation, STOCSY rework |
| H | Group tests boxplot with clickable spectrum |
| I | Annotator dashboard UX |
| J | Time series |

Later work, without separate plans:
- Fixes: decimal input fix, zoom-keeping pick, label editing and fonts.
- STOCSY: gradient line (library change).
- App pages and identity: landing, projects and about pages; identity; theme switch.

Latest commits: `8203a46` theme switch, `6a69fa8` icon corners, `5297782` identity, `e9b6e14` pages, `b0272bc` STOCSY gradient line (metbit library).
