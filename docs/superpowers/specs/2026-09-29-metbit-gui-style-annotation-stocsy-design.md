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

1. **Phase E**: styling (model, endpoints, native args + fallback, style panel,
   per-stage overrides, project persistence).
2. **Phase F**: annotation (model, endpoints, Annotate stage, overlay, popover,
   label list, JSON import / export, project persistence).
3. **Phase G**: STOCSY rework (pre-views, click-to-anchor, clear, cache,
   recent anchors).
