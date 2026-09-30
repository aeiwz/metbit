# metbit GUI Phase C (Bruker import and Process) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users start from raw Bruker FID experiments: pick a folder plus a metadata CSV, then run a Process stage (zero-fill, FFT, phasing, baseline, calibration) with per-sample progress, producing the same matrix metbit's `nmr_preprocessing` produces.

**Architecture:** `ImportParams` gains `source: csv | bruker` with fields shown conditionally (`show_if`). A Bruker import only lists experiments and joins metadata (`BrukerSource`); the new `process` data stage runs `nmr_preprocessing` once per experiment (calibration off) for progress, stacks rows by position with the last experiment's ppm axis exactly as metbit does, then calibrates the whole matrix with metbit's `calibrate`. This was verified to be identical to `nmr_preprocessing(folder)` on the MTBLS2052 heart subset. A new `Stage.blocked` hook disables Process after a CSV import. Spec: `GUI/docs/specs/2026-09-29-metbit-gui-design.md`.

**Tech Stack:** Same as Phase B. New calls: `metbit.nmr.preprocess.nmr_preprocessing`, `metbit.nmr.calibrate.calibrate`. Electron `dialog.showOpenDialog` with `openDirectory`.

## Global Constraints

- All GUI code lives under `GUI/`; Python GUI tests under `tests/gui/`.
- Never use the em dash character or the section sign anywhere.
- Commit messages carry no `Co-Authored-By` trailer.
- Keep every source file under 500 lines.
- Data stage order (exact): `import`, `process`, `normalize`, `align`; `DATA_ORDER` equals this list.
- Working matrix: newest `done` data stage whose output is a `Dataset` (a Bruker import is not).
- Tests that need MTBLS2052 skip when `Benchmark/data/MTBLS2052/proc_subset_heart_wk6` is absent (it is local-only, not in git).
- Python GUI tests: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`. Renderer: `cd GUI && npm run typecheck && npm test`. E2E: `cd GUI && npm run e2e`.

## File Map

```
GUI/server/metbit_server/
  stages/base.py        REPLACE: DATA_ORDER with process, working() Bruker message, Stage.blocked, not_run_figure
  stages/data.py        REPLACE: source-aware Import, BrukerSource, find_experiments, Process
  stages/analysis.py    MODIFY: import not_run_figure from base
  stages/__init__.py    REPLACE: export ProcessParams, BrukerSource
  app.py                MODIFY: "blocked" in listing, 409 on blocked run
tests/gui/
  test_gui_bruker_import.py   NEW
  test_gui_bruker_process.py  NEW
  (five existing files)       MODIFY: expected dependency lists gain "process"
GUI/electron/main.ts, preload.ts      MODIFY: folder picker
GUI/renderer/types.d.ts, api.ts       MODIFY: openFile(directory), blocked, show_if
GUI/renderer/components/ParamPanel.tsx (+ test)  MODIFY: show_if, checkbox, dir-path
GUI/renderer/components/Rail.tsx (+ test)        MODIFY: blocked stages disabled with a tooltip
GUI/renderer/theme.css                MODIFY: checkbox style
GUI/e2e/bruker.spec.ts                NEW (skips without data)
```

---

### Task 1: Source-aware Import and Bruker experiment discovery

**Files:**
- Modify (replace): `GUI/server/metbit_server/stages/base.py`, `GUI/server/metbit_server/stages/data.py`, `GUI/server/metbit_server/stages/__init__.py`
- Modify: `GUI/server/metbit_server/stages/analysis.py` (take `not_run_figure` from base)
- Test: `tests/gui/test_gui_bruker_import.py`

**Interfaces:**
- Consumes: Phase B `base.py` / `data.py`.
- Produces:
  - `base.not_run_figure(message) -> go.Figure` (moved from `analysis.py`; `analysis.py` re-imports it).
  - `base.Stage.blocked: Optional[Callable[[Session], Optional[str]]] = None` (reason string when the stage cannot run).
  - `base.working()` raises `ValueError("Run Process first: Bruker data must be processed before analysis")` when import is done but no data stage holds a `Dataset`.
  - `ImportParams(source="csv" | "bruker", path="", folder="", metadata_path="", sample_column="Sample", group_column="Group")`; validator messages "Choose a CSV file" / "Choose the Bruker folder and a metadata CSV"; schema extras: `path` `{"format": "file-path", "show_if": {"source": "csv"}}`, `folder` `{"format": "dir-path", "show_if": {"source": "bruker"}}`, `metadata_path` `{"format": "file-path", "show_if": {"source": "bruker"}}`, `sample_column` `{"show_if": {"source": "bruker"}}`.
  - `data.find_experiments(folder: str) -> list[tuple[str, str]]` (name, path), searching `fid`, `*/fid`, `*/*/fid` like metbit, sorted by name; raises "No Bruker experiments (folders containing 'fid') found in <folder>".
  - `data.BrukerSource(experiments, meta, labels)`; `meta` rows in experiment order.

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_bruker_import.py`:

```python
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from metbit_server.stages import STAGES, ImportParams
from metbit_server.stages.base import working
from metbit_server.stages.data import BrukerSource, find_experiments

BY_ID = {s.id: s for s in STAGES}


def make_tree(root: Path, names, depth=1):
    for n in names:
        exp = (root / "batch" / n) if depth == 2 else (root / n)
        exp.mkdir(parents=True, exist_ok=True)
        (exp / "fid").write_bytes(b"")
    return root


@pytest.fixture
def bruker(tmp_path):
    folder = make_tree(tmp_path / "exps", ["61030601", "61020601", "61040601"])
    meta = pd.DataFrame({
        "fid": [61020601, 61030601, 61040601, 99],
        "Group": ["sham", "nx", "nx", "x"],
        "Week": [6, 6, 6, 1],
    })
    meta_path = tmp_path / "map.csv"
    meta.to_csv(meta_path, index=False)
    return {"source": "bruker", "folder": str(folder), "metadata_path": str(meta_path), "sample_column": "fid"}


def test_find_experiments_sorted_at_depth_one_and_two(tmp_path):
    make_tree(tmp_path / "one", ["b", "a"])
    assert [n for n, _ in find_experiments(str(tmp_path / "one"))] == ["a", "b"]
    make_tree(tmp_path / "two", ["y", "x"], depth=2)
    assert [n for n, _ in find_experiments(str(tmp_path / "two"))] == ["x", "y"]
    assert find_experiments(str(tmp_path / "one" / "a")) == [("a", str(tmp_path / "one" / "a"))]
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No Bruker experiments"):
        find_experiments(str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="Folder not found"):
        find_experiments(str(tmp_path / "missing"))


def test_bruker_import_joins_metadata_in_experiment_order(run_all, bruker):
    src = run_all(("import", bruker)).output("import")
    assert isinstance(src, BrukerSource)
    assert [n for n, _ in src.experiments] == ["61020601", "61030601", "61040601"]
    assert list(src.meta["fid"]) == [61020601, 61030601, 61040601]
    assert list(src.labels) == ["sham", "nx", "nx"]
    assert list(BY_ID["import"].table(src).columns) == ["fid", "Group", "Week"]
    assert "Run Process" in BY_ID["import"].figures["spectra"](src).layout.annotations[0].text


def test_bruker_import_reports_bad_metadata(run_all, bruker, tmp_path):
    pd.DataFrame({"fid": [61020601], "Group": ["sham"]}).to_csv(tmp_path / "short.csv", index=False)
    with pytest.raises(ValueError, match=r"2 experiment\(s\) have no metadata row"):
        run_all(("import", {**bruker, "metadata_path": str(tmp_path / "short.csv")}))
    with pytest.raises(ValueError, match="Metadata column 'Sample' not found"):
        run_all(("import", {**bruker, "sample_column": "Sample"}))


def test_import_params_require_the_right_paths():
    with pytest.raises(ValidationError, match="Choose a CSV file"):
        ImportParams()
    with pytest.raises(ValidationError, match="Bruker folder and a metadata CSV"):
        ImportParams(source="bruker", folder="/x")
    props = ImportParams.model_json_schema()["properties"]
    assert props["path"]["format"] == "file-path"
    assert props["folder"]["format"] == "dir-path"
    assert props["folder"]["show_if"] == {"source": "bruker"}


def test_analyses_wait_for_process_after_bruker_import(run_all, bruker):
    s = run_all(("import", bruker))
    with pytest.raises(ValueError, match="Run Process first"):
        working(s)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_bruker_import.py --no-cov -q -p no:warnings`
Expected: FAIL with `ImportError: cannot import name 'BrukerSource'`.

- [ ] **Step 3: Base changes**

Replace `GUI/server/metbit_server/stages/base.py` with:

```python
"""Shared stage types and helpers for reading the working matrix."""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel

from metbit.viz.spectra import SpectraPlot

from ..session import Session

Progress = Callable[[float], None]

DATA_ORDER = ["import", "process", "normalize", "align"]
MAX_PLOT_POINTS = 8000


@dataclass
class Stage:
    id: str
    title: str
    params: type[BaseModel]
    run: Callable[[Any, Session, Progress], Any]
    section: str = "Analysis"
    table: Optional[Callable[[Any], pd.DataFrame]] = None
    figures: dict[str, Callable[[Any], go.Figure]] = field(default_factory=dict)
    requires: Optional[str] = None
    after: list[str] = field(default_factory=list)
    # view name -> where a clicked point keeps its ppm: "x" or "customdata" (index 0)
    ppm_views: dict[str, str] = field(default_factory=dict)
    # reason the stage cannot run in this session (shown on the disabled rail item), or None
    blocked: Optional[Callable[[Session], Optional[str]]] = None


@dataclass
class Dataset:
    X: pd.DataFrame                        # samples x features, float ppm column labels
    meta: pd.DataFrame                     # every non-ppm column
    labels: pd.Series                      # group name per sample (str)
    before: Optional[pd.DataFrame] = None  # input matrix, for before / after views


def not_run_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def working(session: Session, upto: Optional[str] = None) -> Dataset:
    """Newest `done` data stage holding a matrix, considering only stages earlier than `upto`."""
    order = DATA_ORDER if upto is None else DATA_ORDER[: DATA_ORDER.index(upto)]
    for sid in reversed(order):
        out = session.outputs.get(sid)
        if session.status.get(sid) == "done" and isinstance(out, Dataset):
            return out
    if session.status.get("import") == "done":
        raise ValueError("Run Process first: Bruker data must be processed before analysis")
    raise ValueError("Import data first")


def groups(session: Session) -> pd.Series:
    return session.output("import").labels


def two_groups(session: Session, a: str, b: str) -> tuple[pd.DataFrame, pd.Series]:
    """Working matrix and labels restricted to groups `a` and `b`."""
    if a == b:
        raise ValueError("Choose two different groups")
    g = groups(session)
    missing = [x for x in (a, b) if x not in set(g)]
    if missing:
        raise ValueError(f"Unknown group(s): {missing}. Groups: {sorted(set(g))}")
    mask = g.isin([a, b]).to_numpy()
    X = working(session).X
    return X[mask].reset_index(drop=True), g[mask].reset_index(drop=True)


def spectra_figure(X: pd.DataFrame, labels: pd.Series, title: str) -> go.Figure:
    # ponytail: display-only decimation keeps figure JSON small; full data stays in the session
    step = max(1, X.shape[1] // MAX_PLOT_POINTS)
    return SpectraPlot(X.iloc[:, ::step], label=list(labels)).overlay(title=title)


def _numeric_columns(meta: pd.DataFrame) -> list[str]:
    return [str(c) for c in meta.columns if pd.to_numeric(meta[c], errors="coerce").notna().all()]


def option_values(session: Session, kind: str) -> Optional[list[str]]:
    if session.status.get("import") != "done":
        return None
    meta = session.output("import").meta
    if kind == "groups":
        return sorted(set(groups(session)))
    if kind == "columns":
        return [str(c) for c in meta.columns]
    if kind == "numeric_columns":
        return _numeric_columns(meta) or None
    return None


def _flatten_nullable(prop: dict) -> None:
    """Optional[X] arrives as anyOf [X, null]; give the form the plain X schema."""
    options = prop.get("anyOf")
    if options and len(options) == 2 and {"type": "null"} in options:
        prop.update(next(o for o in options if o != {"type": "null"}))
        del prop["anyOf"]


def resolve_schema(stage: Stage, session: Session) -> dict:
    """Param JSON schema with data-driven `enum` and `default` for tagged fields."""
    schema = stage.params.model_json_schema()
    for prop in schema.get("properties", {}).values():
        _flatten_nullable(prop)
        values = option_values(session, prop.get("options", ""))
        if values:
            prop["enum"] = values
            if prop.get("default") not in values:
                prop["default"] = values[min(prop.get("pick", 0), len(values) - 1)]
    return schema
```

In `GUI/server/metbit_server/stages/analysis.py`, delete the local `not_run_figure` function (the 6 lines starting `def not_run_figure(message: str) -> go.Figure:`) and change the base import line from:

```python
from .base import DATA_ORDER, Progress, Stage, groups, two_groups, working
```

to:

```python
from .base import DATA_ORDER, Progress, Stage, groups, not_run_figure, two_groups, working
```

- [ ] **Step 4: Source-aware Import**

Replace `GUI/server/metbit_server/stages/data.py` with:

```python
"""Data stages: Import (CSV or Bruker), Normalize, Align."""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from metbit.nmr.alignment import icoshift_align
from metbit.preprocessing.normalize import Normalization

from ..session import Session
from .base import Dataset, Progress, Stage, not_run_figure, spectra_figure, working


# Import ---------------------------------------------------------------------

def _when(source: str) -> dict:
    return {"show_if": {"source": source}}


class ImportParams(BaseModel):
    source: Literal["csv", "bruker"] = Field("csv", title="Source")
    path: str = Field(
        "",
        title="CSV file",
        description="Samples in rows, ppm values as column headers.",
        json_schema_extra={"format": "file-path", **_when("csv")},
    )
    folder: str = Field(
        "",
        title="Bruker folder",
        description="One folder per experiment, each containing a 'fid' file.",
        json_schema_extra={"format": "dir-path", **_when("bruker")},
    )
    metadata_path: str = Field("", title="Metadata CSV", json_schema_extra={"format": "file-path", **_when("bruker")})
    sample_column: str = Field(
        "Sample",
        title="Sample column",
        description="Metadata column holding the experiment folder names.",
        json_schema_extra=_when("bruker"),
    )
    group_column: str = Field("Group", title="Group column")

    @model_validator(mode="after")
    def _paths_given(self) -> "ImportParams":
        if self.source == "csv" and not self.path:
            raise ValueError("Choose a CSV file")
        if self.source == "bruker" and not (self.folder and self.metadata_path):
            raise ValueError("Choose the Bruker folder and a metadata CSV")
        return self


@dataclass
class BrukerSource:
    experiments: list[tuple[str, str]]  # (folder name, path), sorted like metbit
    meta: pd.DataFrame                  # one metadata row per experiment, same order
    labels: pd.Series


def find_experiments(folder: str) -> list[tuple[str, str]]:
    """(name, path) for every experiment, searched like metbit: folder, one or two levels down."""
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"Folder not found: {folder}")
    for pattern in ("fid", "*/fid", "*/*/fid"):
        fids = sorted(root.glob(pattern), key=lambda f: f.parent.name)
        if fids:
            return [(f.parent.name, str(f.parent)) for f in fids]
    raise ValueError(f"No Bruker experiments (folders containing 'fid') found in {folder}")


def _is_ppm(name: Any) -> bool:
    try:
        float(name)
        return True
    except (TypeError, ValueError):
        return False


def _import_csv(p: ImportParams) -> Dataset:
    df = pd.read_csv(p.path)
    feats = [c for c in df.columns if _is_ppm(c)]
    if not feats:
        raise ValueError("No spectral columns found: expected numeric (ppm) column headers")
    meta = df.drop(columns=feats)
    if p.group_column not in meta.columns:
        raise ValueError(f"Group column '{p.group_column}' not found. Columns: {list(meta.columns)}")
    X = df[feats].astype(float)
    if X.isnull().values.any():
        raise ValueError("Spectral matrix contains missing values")
    X.columns = X.columns.astype(float)
    return Dataset(X=X, meta=meta, labels=meta[p.group_column].astype(str).reset_index(drop=True))


def _import_bruker(p: ImportParams) -> BrukerSource:
    experiments = find_experiments(p.folder)
    meta = pd.read_csv(p.metadata_path)
    for col in (p.sample_column, p.group_column):
        if col not in meta.columns:
            raise ValueError(f"Metadata column '{col}' not found. Columns: {list(meta.columns)}")
    keyed = meta.assign(_key=meta[p.sample_column].astype(str)).drop_duplicates("_key").set_index("_key")
    names = [n for n, _ in experiments]
    missing = [n for n in names if n not in keyed.index]
    if missing:
        raise ValueError(
            f"{len(missing)} experiment(s) have no metadata row, for example {missing[:5]}. "
            f"Check the sample column ('{p.sample_column}')."
        )
    matched = keyed.loc[names].reset_index(drop=True)
    return BrukerSource(experiments=experiments, meta=matched, labels=matched[p.group_column].astype(str))


def run_import(p: ImportParams, session: Session, progress: Progress) -> Union[Dataset, BrukerSource]:
    out = _import_bruker(p) if p.source == "bruker" else _import_csv(p)
    progress(1.0)
    return out


def _import_spectra(d: Union[Dataset, BrukerSource]):
    if isinstance(d, BrukerSource):
        return not_run_figure("Run Process to see the spectra.")
    return spectra_figure(d.X, d.labels, "Spectra")


# Normalize ------------------------------------------------------------------

NORMALIZERS = {
    "pqn": "pqn_normalization",
    "msc": "msc_normalization",
    "snv": "snv_normalization",
    "snv_pqn": "snv_pqn_normalization",
}


class NormalizeParams(BaseModel):
    method: Literal["pqn", "msc", "snv", "snv_pqn"] = Field("pqn", title="Method")


def run_normalize(p: NormalizeParams, session: Session, progress: Progress) -> Dataset:
    src = working(session, upto="normalize")
    X = getattr(Normalization(), NORMALIZERS[p.method])(src.X)
    X = pd.DataFrame(X.to_numpy(), index=src.X.index, columns=src.X.columns)  # keep float ppm labels
    progress(1.0)
    return Dataset(X=X, meta=src.meta, labels=src.labels, before=src.X)


# Align ----------------------------------------------------------------------

_WINDOW = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_windows(text: str) -> list[tuple[float, float]]:
    """'3.2-2.8, 1.2-1.5' -> [(3.2, 2.8), (1.5, 1.2)] as (high, low) pairs."""
    out = []
    for part in filter(None, (s.strip() for s in text.split(","))):
        m = _WINDOW.match(part)
        if not m:
            raise ValueError(f"Cannot read window '{part}': use start-end, for example 3.2-2.8")
        a, b = float(m.group(1)), float(m.group(2))
        out.append((max(a, b), min(a, b)))
    return out


class AlignParams(BaseModel):
    windows: str = Field(
        "",
        title="Windows (ppm)",
        description="Comma-separated ranges such as 3.2-2.8, 1.5-1.2. Empty aligns the whole spectrum.",
    )
    reference: Literal["median", "mean"] = Field("median", title="Reference")
    max_shift_ppm: float = Field(0.02, gt=0, le=0.5, title="Max shift (ppm)")

    @field_validator("windows")
    @classmethod
    def _windows_readable(cls, v: str) -> str:
        parse_windows(v)
        return v


def run_align(p: AlignParams, session: Session, progress: Progress) -> Dataset:
    src = working(session, upto="align")
    ppm = src.X.columns.to_numpy(dtype=float)
    windows = parse_windows(p.windows) or [(float(ppm.max()), float(ppm.min()))]
    X, _ = icoshift_align(src.X, ppm, windows=windows, reference=p.reference, max_shift_ppm=p.max_shift_ppm)
    progress(1.0)
    return Dataset(X=X, meta=src.meta, labels=src.labels, before=src.X)


def _before_after(title: str) -> dict:
    return {
        "spectra": lambda d: spectra_figure(d.X, d.labels, title),
        "before": lambda d: spectra_figure(d.before, d.labels, "Before"),
    }


DATA_STAGES: list[Stage] = [
    Stage(
        id="import",
        title="Import",
        section="Data",
        params=ImportParams,
        run=run_import,
        table=lambda d: d.meta,
        figures={"spectra": _import_spectra},
        ppm_views={"spectra": "x"},
    ),
    Stage(
        id="normalize",
        title="Normalize",
        section="Data",
        params=NormalizeParams,
        run=run_normalize,
        figures=_before_after("Normalized"),
        requires="import",
        after=["import"],
        ppm_views={"spectra": "x", "before": "x"},
    ),
    Stage(
        id="align",
        title="Align",
        section="Data",
        params=AlignParams,
        run=run_align,
        figures=_before_after("Aligned"),
        requires="import",
        after=["import", "normalize"],
        ppm_views={"spectra": "x", "before": "x"},
    ),
]
```

Replace `GUI/server/metbit_server/stages/__init__.py` with:

```python
"""Stage registry, in rail order."""
from .analysis import ANALYSIS_STAGES, OPLSParams, OPLSResult, PCAParams, PCAResult, STOCSYParams
from .base import DATA_ORDER, Dataset, Progress, Stage, resolve_schema
from .data import DATA_STAGES, AlignParams, BrukerSource, ImportParams, NormalizeParams, parse_windows
from .multivariate import MULTIVARIATE_STAGES, HCAParams, ICAParams, LDAParams, PLSRParams
from .stats import STATS_STAGES, GroupTestParams, GroupTestResult, VolcanoParams, bin_matrix

_analysis = {s.id: s for s in ANALYSIS_STAGES}

STAGES: list[Stage] = [
    *DATA_STAGES,
    _analysis["pca"],
    _analysis["opls"],
    *MULTIVARIATE_STAGES,
    _analysis["stocsy"],
    *STATS_STAGES,
]
```

`DATA_ORDER` already lists `process`, so existing stages that use `after=list(DATA_ORDER)` now also depend on it. Update the five existing tests that pin those lists, from the repo root:

```bash
python3 - <<'EOF'
import pathlib
edits = {
    "tests/gui/test_gui_stages.py": [('["import", "normalize", "align"]', '["import", "process", "normalize", "align"]')],
    "tests/gui/test_gui_multivariate.py": [('["import", "normalize", "align"]', '["import", "process", "normalize", "align"]')],
}
for path, pairs in edits.items():
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a); s = s.replace(a, b)
    p.write_text(s)
EOF
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `71 passed` (66 + 5 new).

- [ ] **Step 6: Commit**

```bash
git add GUI/server tests/gui
git commit -m "feat(gui): import Bruker experiment folders with a metadata CSV"
```

---

### Task 2: Process stage with per-sample progress

**Files:**
- Modify: `GUI/server/metbit_server/stages/data.py` (add Process, update dependencies)
- Modify: `GUI/server/metbit_server/stages/__init__.py` (export `ProcessParams`)
- Modify: `GUI/server/metbit_server/app.py` (blocked stages)
- Modify: `tests/gui/test_gui_data_stages.py`, `tests/gui/test_gui_group_tests.py`, `tests/gui/test_gui_volcano.py` (registry now includes `process`)
- Test: `tests/gui/test_gui_bruker_process.py`

**Interfaces:**
- Consumes: `BrukerSource`, `find_experiments`, `Stage.blocked`, `not_run_figure` from Task 1.
- Produces:
  - `ProcessParams(bin_size: float = 0.0003 (0 < x <= 0.01), auto_phasing: bool = False, baseline: "none" | "corrector" | "constant" | "explicit" | "median" | "solvent filter" = "none", calibration: "tsp" | "acetate" | "glucose" | "alanine" | "formate" | "none" = "tsp")`.
  - `data._process_one(path: str, p: ProcessParams) -> tuple[np.ndarray, np.ndarray]` (values, ppm) for one experiment; tests monkeypatch it.
  - Stage `process` / `Process`, section `Data`, `requires="import"`, `after=["import"]`, view `spectra` (ppm x), no table, `blocked` reason `PROCESS_CSV_REASON = "Process applies to Bruker imports only. CSV data is already processed."`.
  - `normalize.after == ["import", "process"]`, `align.after == ["import", "process", "normalize"]`.
  - Progress: one event per experiment (fraction of 0.95) then `1.0`.
  - HTTP: `/stages` items gain `blocked: str | null`; `POST /stages/{id}/run` on a blocked stage returns 409 with the reason.

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_bruker_process.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app
from metbit_server.stages import STAGES, ProcessParams
from metbit_server.stages import data as data_mod
from metbit_server.stages.base import working

BY_ID = {s.id: s for s in STAGES}
HEART = Path(__file__).resolve().parents[2] / "Benchmark" / "data" / "MTBLS2052" / "proc_subset_heart_wk6"
SAMPLE_MAP = HEART.parent / "noesy_sample_map.csv"
needs_data = pytest.mark.skipif(not HEART.is_dir(), reason="MTBLS2052 subset is local-only benchmark data")


@pytest.fixture
def bruker(tmp_path):
    for n in ["61030601", "61020601", "61040601"]:
        (tmp_path / "exps" / n).mkdir(parents=True)
        (tmp_path / "exps" / n / "fid").write_bytes(b"")
    meta = pd.DataFrame({"fid": [61020601, 61030601, 61040601], "Group": ["sham", "nx", "nx"]})
    meta.to_csv(tmp_path / "map.csv", index=False)
    return {"source": "bruker", "folder": str(tmp_path / "exps"), "metadata_path": str(tmp_path / "map.csv"), "sample_column": "fid"}


@pytest.fixture
def heart_subset(tmp_path):
    """First six heart experiments plus their metadata rows."""
    names = sorted(d.name for d in HEART.iterdir() if (d / "fid").exists())[:6]
    subset = tmp_path / "heart"
    subset.mkdir()
    for n in names:
        (subset / n).symlink_to(HEART / n)
    m = pd.read_csv(SAMPLE_MAP)
    meta = tmp_path / "heart_meta.csv"
    m[m["fid"].astype(str).isin(names)].to_csv(meta, index=False)
    return {
        "source": "bruker",
        "folder": str(subset),
        "metadata_path": str(meta),
        "sample_column": "fid",
        "group_column": "Factor Value[nephrectomy]",
    }


def test_registry_and_dependencies():
    assert [s.id for s in STAGES][:4] == ["import", "process", "normalize", "align"]
    assert BY_ID["process"].after == ["import"]
    assert BY_ID["normalize"].after == ["import", "process"]
    assert BY_ID["align"].after == ["import", "process", "normalize"]
    assert BY_ID["pca"].after == ["import", "process", "normalize", "align"]


def test_process_stacks_experiments_with_per_sample_progress(run_all, bruker, monkeypatch):
    calls = []

    def one(path, p):
        calls.append(Path(path).name)
        return np.full(50, float(len(calls))), np.linspace(10, -0.5, 50)

    monkeypatch.setattr(data_mod, "_process_one", one)
    s = run_all(("import", bruker))
    seen = []
    out = BY_ID["process"].run(ProcessParams(calibration="none"), s, seen.append)
    assert calls == ["61020601", "61030601", "61040601"]
    assert out.X.shape == (3, 50) and list(out.X.index) == calls
    assert out.X.iloc[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert out.X.columns.dtype == float
    assert len(seen) == 4 and seen == sorted(seen) and seen[-1] == 1.0
    assert list(out.labels) == ["sham", "nx", "nx"]
    s.complete("process", {}, out)
    assert working(s) is out


def test_process_rejects_mismatched_sizes(run_all, bruker, monkeypatch):
    sizes = iter([50, 50, 40])

    def one(path, p):
        n = next(sizes)
        return np.zeros(n), np.linspace(10, 0, n)

    monkeypatch.setattr(data_mod, "_process_one", one)
    with pytest.raises(ValueError, match="same sweep width"):
        run_all(("import", bruker), ("process", {"calibration": "none"}))


def test_process_is_blocked_after_csv_import(run_all, csv_path):
    assert "Bruker" in BY_ID["process"].blocked(run_all())
    c = TestClient(create_app(TOKEN))
    job = c.post("/stages/import/run", json={"path": csv_path}, headers=AUTH).json()["job_id"]
    c.get(f"/jobs/{job}/events", headers=AUTH)
    stages = {x["id"]: x for x in c.get("/stages", headers=AUTH).json()}
    assert "Bruker" in stages["process"]["blocked"]
    assert stages["pca"]["blocked"] is None
    r = c.post("/stages/process/run", json={}, headers=AUTH)
    assert r.status_code == 409 and "Bruker" in r.json()["detail"]


def test_process_params_schema():
    props = ProcessParams.model_json_schema()["properties"]
    assert props["auto_phasing"]["type"] == "boolean"
    assert props["baseline"]["default"] == "none"


@needs_data
def test_process_matches_metbit_on_real_bruker_data(run_all, heart_subset):
    from metbit.nmr.preprocess import nmr_preprocessing

    out = run_all(("import", heart_subset), ("process", {})).output("process")
    ref = nmr_preprocessing(heart_subset["folder"]).get_data()
    assert out.X.shape == ref.shape
    assert np.allclose(out.X.to_numpy(), ref.to_numpy())
    assert np.allclose(out.X.columns.to_numpy(float), ref.columns.to_numpy(float))
    assert list(out.X.index) == [str(i) for i in ref.index]


@needs_data
def test_real_bruker_pipeline_over_http(heart_subset):
    c = TestClient(create_app(TOKEN))

    def run(sid, params):
        job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
        return c.get(f"/jobs/{job}/events", headers=AUTH).text

    assert "event: done" in run("import", heart_subset)
    events = run("process", {})
    assert events.count("event: progress") == 7 and "event: done" in events
    assert "event: done" in run("pca", {})
    assert c.get("/stages/process/figure?view=spectra", headers=AUTH).status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_bruker_process.py --no-cov -q -p no:warnings`
Expected: FAIL with `ImportError: cannot import name 'ProcessParams'`.

- [ ] **Step 3: Add the Process stage**

In `GUI/server/metbit_server/stages/data.py`:

Change the imports block top to:

```python
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from metbit.nmr.alignment import icoshift_align
from metbit.nmr.calibrate import calibrate
from metbit.nmr.preprocess import nmr_preprocessing
from metbit.preprocessing.normalize import Normalization
```

Insert this block immediately before the line `# Normalize ---...`:

```python
# Process (Bruker) -------------------------------------------------------------

PROCESS_CSV_REASON = "Process applies to Bruker imports only. CSV data is already processed."


class ProcessParams(BaseModel):
    bin_size: float = Field(
        0.0003,
        gt=0,
        le=0.01,
        title="Point spacing (ppm)",
        description="Zero-filling target. 0.0003 gives about 67k points over a 20 ppm window.",
    )
    auto_phasing: bool = Field(
        False, title="Automatic phasing", description="Off uses the phase stored with each experiment (PHC0 / PHC1)."
    )
    baseline: Literal["none", "corrector", "constant", "explicit", "median", "solvent filter"] = Field(
        "none", title="Baseline correction"
    )
    calibration: Literal["tsp", "acetate", "glucose", "alanine", "formate", "none"] = Field(
        "tsp", title="Calibration", description="TSP also covers DSS: both reference 0 ppm."
    )


def _process_one(path: str, p: ProcessParams) -> tuple[np.ndarray, np.ndarray]:
    """metbit's own pipeline for one experiment, uncalibrated: (values, ppm)."""
    d = nmr_preprocessing(
        path,
        bin_size=p.bin_size,
        auto_phasing=p.auto_phasing,
        baseline_correction=p.baseline != "none",
        baseline_type=p.baseline,
        calibration=False,
    ).get_data()
    return d.to_numpy()[0], d.columns.to_numpy(dtype=float)


def run_process(p: ProcessParams, session: Session, progress: Progress) -> Dataset:
    src = session.output("import")
    if not isinstance(src, BrukerSource):
        raise ValueError(PROCESS_CSV_REASON)
    rows: list[np.ndarray] = []
    ppm: Optional[np.ndarray] = None
    for i, (name, path) in enumerate(src.experiments):
        values, ppm = _process_one(path, p)
        if rows and len(values) != len(rows[0]):
            raise ValueError(
                f"Experiment {name} has {len(values)} points but earlier ones have {len(rows[0])}. "
                "metbit stacks spectra by position, so all experiments need the same sweep width."
            )
        rows.append(values)
        progress(0.95 * (i + 1) / len(src.experiments))
    # Same as metbit's nmr_preprocessing on the whole folder: rows stacked by position under the
    # last experiment's ppm axis, then calibrated together (verified identical on MTBLS2052).
    X = pd.DataFrame(np.vstack(rows), index=[n for n, _ in src.experiments], columns=ppm)
    if p.calibration != "none":
        X = calibrate(X, ppm, p.calibration)
    X.columns = X.columns.astype(float)
    progress(1.0)
    return Dataset(X=X, meta=src.meta, labels=src.labels)


def _process_blocked(session: Session) -> Optional[str]:
    if session.status.get("import") == "done" and not isinstance(session.outputs.get("import"), BrukerSource):
        return PROCESS_CSV_REASON
    return None


```

In the `DATA_STAGES` list, insert this stage between the `import` and `normalize` entries:

```python
    Stage(
        id="process",
        title="Process",
        section="Data",
        params=ProcessParams,
        run=run_process,
        figures={"spectra": lambda d: spectra_figure(d.X, d.labels, "Processed spectra")},
        requires="import",
        after=["import"],
        ppm_views={"spectra": "x"},
        blocked=_process_blocked,
    ),
```

and change `after=["import"],` in the `normalize` entry to `after=["import", "process"],`, and `after=["import", "normalize"],` in the `align` entry to `after=["import", "process", "normalize"],`.

In `GUI/server/metbit_server/stages/__init__.py`, change the data import line to:

```python
from .data import DATA_STAGES, AlignParams, BrukerSource, ImportParams, NormalizeParams, ProcessParams, parse_windows
```

- [ ] **Step 4: Blocked stages in the API**

In `GUI/server/metbit_server/app.py`, add to the dict in `list_stages` after `"has_table": s.table is not None,`:

```python
                "blocked": s.blocked(session) if s.blocked else None,
```

and in `run`, after the `requires` check (`raise HTTPException(409, f"Run '{stage.requires}' first")`), add:

```python
        reason = stage.blocked(session) if stage.blocked else None
        if reason:
            raise HTTPException(409, reason)
```

- [ ] **Step 5: Update registry expectations in older tests**

Run from the repo root:

```bash
python3 - <<'EOF'
import pathlib
edits = {
    "tests/gui/test_gui_data_stages.py": [
        ('assert [s.id for s in STAGES][:3] == ["import", "normalize", "align"]',
         'assert [s.id for s in STAGES][:4] == ["import", "process", "normalize", "align"]'),
        ('assert BY_ID["normalize"].after == ["import"]', 'assert BY_ID["normalize"].after == ["import", "process"]'),
        ('assert BY_ID["align"].after == ["import", "normalize"]', 'assert BY_ID["align"].after == ["import", "process", "normalize"]'),
    ],
    "tests/gui/test_gui_group_tests.py": [('"import", "normalize", "align",', '"import", "process", "normalize", "align",')],
    "tests/gui/test_gui_volcano.py": [
        ('assert [s.id for s in STAGES][:3] == ["import", "normalize", "align"]',
         'assert [s.id for s in STAGES][:4] == ["import", "process", "normalize", "align"]'),
        ("assert all(s.section == \"Data\" for s in STAGES[:3])", "assert all(s.section == \"Data\" for s in STAGES[:4])"),
    ],
}
for path, pairs in edits.items():
    p = pathlib.Path(path); s = p.read_text()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a); s = s.replace(a, b)
    p.write_text(s)
EOF
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `78 passed` (71 + 7 new) when MTBLS2052 is present; `76 passed, 2 skipped` without it.

- [ ] **Step 7: Commit**

```bash
git add GUI/server tests/gui
git commit -m "feat(gui): add Bruker Process stage with per-sample progress"
```

---

### Task 3: Renderer: folder picker, conditional fields, checkboxes, blocked stages

**Files:**
- Modify: `GUI/electron/main.ts`, `GUI/electron/preload.ts`, `GUI/renderer/types.d.ts`, `GUI/renderer/api.ts`
- Modify (replace): `GUI/renderer/components/ParamPanel.tsx`, `GUI/renderer/components/Rail.tsx`
- Modify: `GUI/renderer/components/ParamPanel.test.tsx`, `GUI/renderer/components/Rail.test.tsx` (append tests), `GUI/renderer/theme.css`

**Interfaces:**
- Consumes: schema extras `show_if`, `format: "dir-path"`, `type: "boolean"`; `/stages` field `blocked`.
- Produces:
  - `window.metbit.openFile(filters?, directory?: boolean)`; `directory` opens a folder picker.
  - `FieldSchema.show_if?: Record<string, unknown>`; `StageInfo.blocked: string | null`.
  - `ParamPanel` hides fields whose `show_if` does not match current values; renders booleans as checkboxes; `dir-path` Browse calls `openFile(undefined, true)`.
  - `Rail` disables stages with a `blocked` reason and shows it as the button `title`.

- [ ] **Step 1: Write the failing tests**

Append to `GUI/renderer/components/ParamPanel.test.tsx`:

```tsx
const importSchema: JsonSchema = {
  properties: {
    source: { type: "string", title: "Source", enum: ["csv", "bruker"], default: "csv" },
    path: { type: "string", title: "CSV file", format: "file-path", show_if: { source: "csv" } },
    folder: { type: "string", title: "Bruker folder", format: "dir-path", show_if: { source: "bruker" } },
    auto_phasing: { type: "boolean", title: "Automatic phasing", default: false },
  },
};

it("shows only the fields whose show_if matches", async () => {
  const { root } = mount({ schema: importSchema });
  expect(root.querySelector("#f-path")).not.toBeNull();
  expect(root.querySelector("#f-folder")).toBeNull();
  const source = root.querySelector<HTMLSelectElement>("#f-source")!;
  source.value = "bruker";
  source.dispatchEvent(new Event("change"));
  await tick();
  expect(root.querySelector("#f-path")).toBeNull();
  expect(root.querySelector("#f-folder")).not.toBeNull();
});

it("renders booleans as checkboxes and submits booleans", async () => {
  const { root, onRun } = mount({ schema: importSchema });
  const box = root.querySelector<HTMLInputElement>("#f-auto_phasing")!;
  expect(box.type).toBe("checkbox");
  box.click();
  await tick();
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onRun.mock.calls[0][0].auto_phasing).toBe(true);
});

it("opens a folder picker for dir-path fields", async () => {
  const openFile = vi.fn().mockResolvedValue("/data/exps");
  (window as any).metbit = { openFile };
  const { root } = mount({ schema: importSchema, initial: { source: "bruker" } });
  [...root.querySelectorAll("button")].find((b) => b.textContent === "Browse")!.click();
  await tick();
  expect(openFile).toHaveBeenCalledWith(undefined, true);
  await tick();
  expect(root.querySelector<HTMLInputElement>("#f-folder")!.value).toBe("/data/exps");
});
```

Append to `GUI/renderer/components/Rail.test.tsx`:

```tsx
it("disables blocked stages and explains why", () => {
  const root = document.createElement("div");
  const process = { ...stage("process", "Data", "import"), blocked: "Bruker only" };
  render(<Rail stages={[stage("import", "Data", null, "done"), process]} current="import" onSelect={() => {}} />, root);
  const button = [...root.querySelectorAll("button")].find((b) => b.textContent === "PROCESS")!;
  expect(button.disabled).toBe(true);
  expect(button.title).toBe("Bruker only");
});
```

In the same file, add `blocked: null,` to the object returned by the `stage` helper (after `has_table: true,`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd GUI && npm test`
Expected: 4 failed (show_if, checkbox, dir-path, blocked).

- [ ] **Step 3: Bridge and types**

In `GUI/electron/main.ts`, replace the `dialog:openFile` handler with:

```ts
ipcMain.handle("dialog:openFile", async (_e, filters?: Electron.FileFilter[], directory?: boolean) => {
  const r = await dialog.showOpenDialog(win!, { properties: [directory ? "openDirectory" : "openFile"], filters });
  return r.canceled ? null : r.filePaths[0];
});
```

In `GUI/electron/preload.ts`, replace the `openFile` line with:

```ts
  openFile: (filters?: { name: string; extensions: string[] }[], directory?: boolean) =>
    ipcRenderer.invoke("dialog:openFile", filters, directory),
```

In `GUI/renderer/types.d.ts`, replace the `openFile` line with:

```ts
    openFile(filters?: { name: string; extensions: string[] }[], directory?: boolean): Promise<string | null>;
```

In `GUI/renderer/api.ts`, add `show_if?: Record<string, unknown>;` after `pick_ppm?: boolean;` in `FieldSchema`, and `blocked: string | null;` after `has_table: boolean;` in `StageInfo`.

- [ ] **Step 4: Parameter panel**

Replace `GUI/renderer/components/ParamPanel.tsx` with:

```tsx
import { useState } from "preact/hooks";
import type { FieldSchema, JsonSchema } from "../api";

export type Values = Record<string, unknown>;
export type FieldErrors = Record<string, string>;

export function defaults(schema: JsonSchema): Values {
  return Object.fromEntries(Object.entries(schema.properties).map(([k, f]) => [k, f.default ?? ""]));
}

export function fieldErrors(detail: unknown): FieldErrors {
  if (!Array.isArray(detail)) return { "": String(detail) };
  return Object.fromEntries(detail.map((d: { loc: unknown[]; msg: string }) => [String(d.loc[0] ?? ""), d.msg]));
}

const visible = (f: FieldSchema, values: Values) =>
  !f.show_if || Object.entries(f.show_if).every(([k, v]) => values[k] === v);

interface Props {
  title: string;
  schema: JsonSchema;
  initial: Values | null;
  errors: FieldErrors;
  busy: boolean;
  onRun: (values: Values) => void;
  picked?: number | null;
}

export function ParamPanel({ title, schema, initial, errors, busy, onRun, picked = null }: Props) {
  const [values, setValues] = useState<Values>(() => ({ ...defaults(schema), ...initial }));
  const set = (k: string, v: unknown) => setValues((s) => ({ ...s, [k]: v }));
  return (
    <form
      class="panel"
      onSubmit={(e) => {
        e.preventDefault();
        onRun(values);
      }}
    >
      <h2>{title}</h2>
      {Object.entries(schema.properties)
        .filter(([, f]) => visible(f, values))
        .map(([k, f]) => (
          <Field key={k} name={k} field={f} value={values[k]} error={errors[k]} picked={picked} onChange={(v) => set(k, v)} />
        ))}
      {errors[""] && <p class="error">{errors[""]}</p>}
      <button class="btn primary" type="submit" disabled={busy}>
        {busy ? "Running…" : "Run"}
      </button>
    </form>
  );
}

interface FieldProps {
  name: string;
  field: FieldSchema;
  value: unknown;
  error?: string;
  picked: number | null;
  onChange: (v: unknown) => void;
}

function Field({ name, field, value, error, picked, onChange }: FieldProps) {
  const id = `f-${name}`;
  const text = (e: Event) => onChange((e.currentTarget as HTMLInputElement).value);
  let control;
  if (field.enum) {
    control = (
      <select id={id} value={String(value)} onChange={text}>
        {field.enum.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    );
  } else if (field.type === "boolean") {
    control = <input id={id} type="checkbox" checked={value === true} onChange={(e) => onChange(e.currentTarget.checked)} />;
  } else if (field.type === "integer" || field.type === "number") {
    control = (
      <input
        id={id}
        type="number"
        step={field.type === "integer" ? 1 : "any"}
        min={field.minimum}
        max={field.maximum}
        value={value as number}
        onInput={(e) => {
          const raw = e.currentTarget.value;
          onChange(raw === "" ? "" : Number(raw));
        }}
      />
    );
  } else if (field.format === "file-path" || field.format === "dir-path") {
    const directory = field.format === "dir-path";
    const browse = async () => {
      // ponytail: every file input so far is a CSV; take filters from the schema when that changes
      const path = await window.metbit.openFile(directory ? undefined : [{ name: "CSV", extensions: ["csv"] }], directory || undefined);
      if (path) onChange(path);
    };
    control = (
      <div class="row">
        <input id={id} value={String(value ?? "")} onInput={text} />
        <button type="button" class="btn" onClick={browse}>
          Browse
        </button>
      </div>
    );
  } else {
    control = <input id={id} value={String(value ?? "")} onInput={text} />;
  }
  return (
    <div class={`field${error ? " invalid" : ""}`}>
      <label for={id}>{field.title ?? name}</label>
      {control}
      {field.pick_ppm && picked !== null && (
        <button type="button" class="pick" onClick={() => onChange(picked)}>
          Use {picked.toFixed(3)} ppm
        </button>
      )}
      {field.description && <span class="hint">{field.description}</span>}
      {error && <span class="err">{error}</span>}
    </div>
  );
}
```

Note: the CSV Browse now calls `openFile(filters, undefined)`; the stage 1 behavior is unchanged.

- [ ] **Step 5: Rail**

Replace `GUI/renderer/components/Rail.tsx` with:

```tsx
import { Fragment } from "preact";
import type { StageInfo } from "../api";

interface Props {
  stages: StageInfo[];
  current: string;
  onSelect: (id: string) => void;
}

export function Rail({ stages, current, onSelect }: Props) {
  const statusOf = Object.fromEntries(stages.map((s) => [s.id, s.status]));
  return (
    <nav class="rail" aria-label="Pipeline">
      <div class="rail-brand">metbit</div>
      {stages.map((s, i) => {
        // reachable when its input exists, or it already has its own result
        const reachable = !s.requires || statusOf[s.requires] === "done" || s.status !== "empty";
        return (
          <Fragment key={s.id}>
            {s.section !== stages[i - 1]?.section && <div class="rail-section">{s.section}</div>}
            <button
              aria-current={s.id === current}
              disabled={!reachable || !!s.blocked}
              title={s.blocked ?? undefined}
              onClick={() => onSelect(s.id)}
            >
              <span class={`dot ${s.status}`} />
              {s.title}
            </button>
          </Fragment>
        );
      })}
    </nav>
  );
}
```

Append to `GUI/renderer/theme.css`:

```css
.field input[type="checkbox"] { width: 16px; height: 16px; margin: 0; accent-color: var(--accent); align-self: flex-start; }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd GUI && npm run typecheck && npm test`
Expected: typecheck exits 0; `17 passed`.

- [ ] **Step 7: Commit**

```bash
git add GUI/electron GUI/renderer
git commit -m "feat(gui): folder picker, conditional fields, checkboxes and blocked stages"
```

---

### Task 4: End-to-end Bruker path

**Files:**
- Create: `GUI/e2e/bruker.spec.ts`

**Interfaces:**
- Consumes: control ids `#f-source`, `#f-folder`, `#f-metadata_path`, `#f-sample_column`, `#f-group_column`; rail titles `Process`, `PCA`.
- Produces: `npm run e2e` runs 4 tests (the Bruker one skips without MTBLS2052).

- [ ] **Step 1: Write the test**

Create `GUI/e2e/bruker.spec.ts`:

```ts
import { _electron as electron, expect, test } from "@playwright/test";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const HEART = join(__dirname, "..", "..", "Benchmark", "data", "MTBLS2052", "proc_subset_heart_wk6");
const MAP = join(HEART, "..", "noesy_sample_map.csv");

test.skip(!existsSync(HEART), "MTBLS2052 subset is local-only benchmark data");

test("Bruker folder to processed spectra to PCA", async () => {
  // the sample map covers every matrix; the join only needs the rows for these experiments
  const meta = join(mkdtempSync(join(tmpdir(), "metbit-e2e-")), "heart_meta.csv");
  writeFileSync(meta, readFileSync(MAP, "utf8"));

  const app = await electron.launch({ args: ["."] });
  const win = await app.firstWindow();
  const run = () => win.getByRole("button", { name: "Run", exact: true }).click();

  await win.selectOption("#f-source", "bruker");
  await win.fill("#f-folder", HEART);
  await win.fill("#f-metadata_path", meta);
  await win.fill("#f-sample_column", "fid");
  await win.fill("#f-group_column", "Factor Value[nephrectomy]");
  await run();
  await expect(win.locator("table.data")).toContainText("61020601");

  await win.locator(".rail button", { hasText: "Process" }).click();
  await run();
  await expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 60_000 });

  await win.locator(".rail button", { hasText: "PCA" }).click();
  await run();
  await expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

  await app.close();
});
```

- [ ] **Step 2: Run everything**

Run from the repo root:
```bash
python3 -m pytest -q -p no:cacheprovider && (cd GUI && npm run typecheck && npm test && npm run e2e)
```
Expected: repo suite passes with coverage at or above 99%; typecheck exits 0; `17 passed`; `4 passed` (3 when MTBLS2052 is absent, with 1 skipped).

- [ ] **Step 3: Commit**

```bash
git add GUI/e2e/bruker.spec.ts
git commit -m "test(gui): cover the Bruker import and Process path end to end"
```
