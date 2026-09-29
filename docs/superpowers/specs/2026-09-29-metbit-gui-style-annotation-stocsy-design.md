# metbit GUI - Plot Styling, Peak Annotation, STOCSY Rework - Design Spec

Date: 2026-09-29
Status: Approved design, pending implementation plans
Builds on: `docs/superpowers/specs/2026-09-29-metbit-gui-design.md` (phases A to D implemented)

## Goals

1. Let users style plots: per-group color and symbol, marker size, opacity,
   font size, line width, ellipses. Global defaults plus per-stage overrides.
2. Bring metbit's peak annotator (`metbit.apps.annotate.annotate_peak`) into the
   GUI: label peaks, see labels on every spectrum, export an intensity table.
3. Rework the STOCSY stage to follow metbit's Dash app
   (`metbit.apps.stocsy_app.STOCSY_app`): pick the anchor by clicking a
   spectrum, run, cache results per anchor.
4. Show, in the Import stage, mock-up visuals of compliant input files and
   step-by-step preparation guides for both raw Bruker + metadata and a
   processed data table.

## 1. Plot styling

### Model (`GUI/server/metbit_server/stages/style.py`)

```
PlotStyle (pydantic):
  groups: dict[str, GroupStyle]      # GroupStyle: color "#rrggbb", symbol
  marker_size: int      4..60   default 12
  marker_opacity: float 0.1..1  default 0.8
  font_size: int        8..32   default 14
  line_width: float     0.5..5  default 1.2
  show_ellipses: bool           default true
```

- Symbols: `circle`, `square`, `diamond`, `triangle-up`, `triangle-down`,
  `cross`, `x`, `star` (Plotly names).
- Default group colors come from a colorblind-safe qualitative palette
  (Okabe-Ito), assigned in sorted group order when groups first appear.
- Colors are validated as `#rrggbb`; unknown groups in a style are ignored.

### Global and per-stage

- The session holds one global `PlotStyle`. `GET /style` returns it with every
  current group filled in; `PUT /style` replaces it (422 on invalid values).
- Every stage with plots accepts an optional `style` object in its params
  (all fields optional). Effective style = global merged with the stage
  override, stage fields winning, group entries merged per key.
- A style change never re-runs an analysis: figures are rendered from the
  stored result with the effective style at request time.
  `PUT /stages/{id}/style` stores a stage override without a job.
- Global style and stage overrides are saved in the `.metbit` project
  (`params.json` gains `style` and each stage entry gains `style`). Version
  stays 1; older projects load with default style.

### Applying style

- Figure functions take `(result, style)` instead of `(result)`.
- Native metbit arguments are used where the plot function supports them:
  `color_dict`, `symbol_` / `symbol_dict`, `marker_size`, `marker_opacity`,
  `font_size`, `individual_ellipse` (PCA and OPLS-DA scores), `color_dict`
  (LDA, ICA, SpectraPlot), `custom_colors` (ANOVA / Kruskal boxplot),
  `marker_size` / `font_size` (VIP), `linewidth` (SpectraPlot).
- `apply_style(fig, style)` then runs on every figure: traces whose `name`
  or `legendgroup` equals a group get that group's color and symbol
  (markers) or color (lines), and the layout font size is set. This covers
  plots whose metbit function lacks an argument.

### UI

- A "Plot style" item at the bottom of the rail (next to Open / Save project)
  opens a style panel in the right column: one row per group with a native
  color input and a symbol select, then the numeric fields and the ellipse
  checkbox. Changes apply on blur / change and the current plot re-fetches.
- Each stage with figures gets a collapsed "Style" section in its parameter
  panel with the same fields, all empty by default meaning "use global".

## 2. Peak annotation

### Model

- Session holds `annotations: list[{id, ppm, label, angle}]` (global).
- `GET /annotations`, `POST /annotations` `{ppm, label, angle?}`,
  `DELETE /annotations/{id}`, `PUT /annotations` (replace all, used by JSON
  import). Changes are immediate, not jobs, and do not stale any stage.
- Labels are sanitized like metbit's annotator: trimmed, at most 256
  characters, HTML-escaped; empty labels are rejected (422).
- Saved in the `.metbit` project (`params.json` gains `annotations`).
- Label JSON import / export uses the same shape as metbit's annotator
  (`x`, `text`, `textangle` accepted on import).

### Annotate stage

- Section `Statistics`, id `annotate`, title `Annotate`, requires import.
- Params: `mode` (`single` | `median`) plus the optional per-stage `style`.
  Intensities are read at the nearest ppm point, as metbit's annotator does
  (no binning).
- Views:
  - `spectra`: working matrix (single: every spectrum, decimated as other
    spectrum views; median: one line per group) with every annotation drawn
    as an arrow label at the nearest point of the displayed curve.
  - Table: intensity table, one row per sample: metadata columns, then one
    column per annotation named `<label>_<ppm to 3 dp>` holding the working
    matrix value at the nearest ppm. CSV cells starting with `= + - @ TAB`
    are prefixed with `'` on export, as metbit does.
- Clicking the spectra view opens a small popover anchored to the click with
  a label field and an angle field; Enter adds the annotation.
- The parameter panel lists current labels (`label @ ppm`) with delete
  buttons, and "Import labels" / "Export labels" actions (JSON).

### Overlay elsewhere

- Every spectrum view with a ppm x-axis (Import / Process / Normalize / Align
  spectra, Annotate spectra, STOCSY spectra and correlation) draws the global
  annotations server-side, so exports include them.
- Outside the Annotate stage a click keeps its current meaning (pick ppm).

### Not included

- metbit's per-sample line style picker: group-level styling covers it.
- HTML export of the annotated plot: PNG / SVG export already exists.

## 3. STOCSY rework

- Stages may declare `pre_views`: views rendered from the working matrix
  before the stage has a result. STOCSY declares `spectra`.
- STOCSY `spectra` view: a seeded random sample of the working spectra,
  like the Dash app (all when 5 or fewer, 20% when 100 or more, else up to
  30), styled and annotated.
- Clicking the STOCSY spectra view sets the Anchor field directly (no
  "Use ppm" step). A "Clear" action beside Run empties it.
- Run computes the correlation (`STOCSY(X, anchor, p)`), adds a
  `correlation` view, and switches to it.
- Results are cached per `(anchor, p_threshold)` in the session; the cache is
  cleared when any data stage completes. A "Recent anchors" list in the
  parameter panel re-runs a cached anchor instantly.

## 4. Import format guide (mock-up file visuals)

### Where

- The Import stage's center view shows the guide whenever Import has no
  result. It follows the form's **Source** field live (CSV or Bruker).
- After data is loaded, a "File format" link in the tab bar reopens it.
- `ParamPanel` gains an `onValuesChange(values)` callback so the App knows the
  current Source.

### Processed data table (Source = CSV)

Mock spreadsheet (styled HTML table, metadata columns and spectral columns
in two tinted bands, with captions):

```
Sample | Group   | Age | 10.000 | 9.997 | 9.994 | ... | 0.000
s01    | Control | 34  | 0.0123 | 0.0119| 0.0121| ... | 0.0045
s02    | Treated | 51  | 0.0131 | 0.0127| 0.0125| ... | 0.0051
```

How to prepare (numbered checklist beside the mock-up):

1. One row per sample, one column per chemical shift. Spectral column headers
   are the ppm values as plain numbers (for example `9.997`); any header that
   parses as a number is read as a spectral column.
2. Metadata columns (sample ID, group, other factors) have non-numeric
   headers and can sit anywhere, usually first.
3. Include a group column; its name goes in the form's Group column field.
   Numeric metadata such as age or dose can later be a PLS-R target.
4. Spectra must already be processed: phased, baseline corrected and
   calibrated (TSP / DSS at 0 ppm). Binned data is fine.
5. No missing values. Remove unwanted regions (for example water) by deleting
   those columns.
6. Save as a comma-separated UTF-8 CSV. From Python:
   `pd.concat([meta, spectra], axis=1).to_csv("data.csv", index=False)`.
   Any data stage's Export CSV in this app writes this format too.

"Save example CSV" writes a synthetic compliant file (12 samples, 2 groups,
about 400 ppm points with a few peaks), generated in the renderer and saved
through the save dialog.

### Raw Bruker + metadata (Source = Bruker)

Two mock-ups side by side, joined by connector lines from the experiment
folder names to the matching metadata cells:

```
study/                              metadata.csv
  61020601/   ----------------->    fid      | Group   | Week
    fid                             61020601 | sham    | 6
    acqus                           61030601 | nx      | 6
    pdata/1/procs
  61030601/   ----------------->
    ...
```

How to prepare:

1. Export each experiment from TopSpin as its own folder containing `fid`,
   `acqus` and `pdata/1/procs` (Process reads the stored phase PHC0 / PHC1 and
   the spectrum offset from `procs`).
2. Put all experiments of one study under a single folder, either directly
   (`study/<exp>/fid`) or one level deeper (`study/<batch>/<exp>/fid`).
3. Acquire every experiment with the same sweep width and spectrometer
   settings: spectra are stacked by position, so point counts must match.
4. Keep an internal reference (TSP or DSS) in each sample if you want
   automatic calibration to 0 ppm.
5. Make a metadata CSV with one row per experiment: a column holding the exact
   experiment folder names (its name goes in Sample column, for example
   `fid`), a group column, and any other factors. Every experiment needs a
   row; extra rows are ignored.

No Bruker example file is offered (raw FIDs cannot be synthesized
meaningfully).

### Tests

- Unit: guide renders the CSV mock-up and checklist by default and the Bruker
  mock-up and checklist after Source changes; the example CSV generator's
  headers pass the importer's ppm check (numeric) and include `Sample` and
  `Group`.
- E2E: save the example CSV via the test seam, import it, see the table.

## 5. Loading animation (molecule)

- `components/MoleculeLoader.tsx`: inline SVG + CSS keyframes, no dependency.
  A benzene ring (six atoms, six bonds, six inner double-bond strokes in two
  alternating Kekule sets) in the accent color on thin strokes. The ring
  turns once every 8 s, a brightness pulse travels around the atoms, and the
  two double-bond sets cross-fade (resonance).
- Caption below (`Starting metbit…`, `Running <stage title>…`,
  `Opening project…`, `Restoring session…`) and a percentage when the backend
  has reported real progress (for example per sample in Process).
- `prefers-reduced-motion: reduce` stops all animation; caption and
  percentage still update. The container is `role="status"`.
- Shown at startup, in the center view while a stage runs, while a project
  opens, and while a session restores after a backend restart. The 2 px
  top bar stays.

## Testing

- Server: style merge (stage beats global), native args reach metbit (trace
  colors in figure JSON), `apply_style` fallback, style validation 422;
  annotation add / delete / sanitize / 422 on empty; project round trip with
  style and annotations; intensity table values and CSV injection guard;
  overlay present on a spectrum figure; STOCSY pre-view without a result,
  cache hit, cache cleared by a data stage.
- Renderer: style panel builds a row per group; annotation popover submits
  label and angle; STOCSY click sets the anchor.
- E2E: set a group color and see it in the PCA scores; annotate a peak and see
  the label on the Normalize spectra; click-to-run STOCSY.

## Build order

0. **Phase E0**: import format guide (mock-ups, preparation checklists,
   example CSV).
1. **Phase E**: styling (model, endpoints, native args + fallback, style panel,
   per-stage overrides, project persistence).
1b. **Phase E1**: molecule loading animation.
2. **Phase F**: annotation (model, endpoints, Annotate stage, overlay, popover,
   label list, JSON import / export, project persistence).
3. **Phase G**: STOCSY rework (pre-views, click-to-anchor, clear, cache,
   recent anchors).
