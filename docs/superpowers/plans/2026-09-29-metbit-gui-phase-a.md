# metbit GUI Phase A (Graph, Normalize, Align, OPLS-DA, STOCSY, Volcano) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the stage 1 app into a real metabolomics workbench: dependency-graph staleness, a sectioned rail, data-driven dropdowns, Normalize and Align data stages, OPLS-DA, STOCSY with click-to-pick ppm, and a Volcano plot.

**Architecture:** The sidecar's single `stages.py` becomes a `stages/` package (`base`, `data`, `analysis`, `stats`). Each `Stage` declares its rail `section`, the stages it depends on (`after`), and which views have a ppm x-axis (`ppm_views`). Analyses read the "working matrix" (newest `done` data stage). The server injects group names into schemas for fields tagged `options: "groups"`. The renderer adds rail sections, field hints, a "Use picked ppm" button, and plot click capture. Spec: `docs/superpowers/specs/2026-09-29-metbit-gui-design.md` (sections "Pipeline model" and "Screens").

**Tech Stack:** Same as stage 1. New metbit calls: `metbit.preprocessing.normalize.Normalization`, `metbit.nmr.alignment.icoshift_align`, `metbit.opls_da`, `metbit.analysis.stocsy.STOCSY`, `metbit.stats.multitest.VolcanoPlot`, `metbit.viz.spectra.SpectraPlot`.

## Global Constraints

- All GUI code lives under `GUI/`; Python GUI tests under `tests/gui/`.
- Never use the em dash character or the section sign anywhere.
- Commit messages carry no `Co-Authored-By` trailer.
- Keep every source file under 500 lines.
- Stage statuses (exact strings): `empty`, `running`, `done`, `failed`, `stale`.
- Rail sections (exact strings, in order): `Data`, `Analysis`, `Statistics`.
- Stage ids (exact, rail order after Phase A): `import`, `normalize`, `align`, `pca`, `opls`, `stocsy`, `volcano`.
- Working matrix order: `align`, then `normalize`, then `import`; only `done` stages count.
- Python GUI tests: `python3 -m pytest tests/gui --no-cov -q`. Renderer: `cd GUI && npm run typecheck && npm test`. E2E: `cd GUI && npm run e2e`.
- Visual language unchanged: accent `#2563EB`, Inter 13 px, 8 px grid, hairline borders, no shadows.

## File Map

```
GUI/server/metbit_server/
  session.py                 MODIFY: `after` graph, transitive staleness
  stages.py                  DELETE (becomes the package below)
  stages/__init__.py         registry STAGES + re-exports
  stages/base.py             Stage, Dataset, working(), groups(), two_groups(), spectra_figure(), resolve_schema()
  stages/data.py             Import, Normalize, Align, parse_windows()
  stages/analysis.py         PCA, OPLS-DA, STOCSY
  stages/stats.py            Volcano
  app.py                     MODIFY: graph Session, listing fields, schema injection, table 404
tests/gui/
  conftest.py                MODIFY: positive intensities, run_all fixture
  test_gui_stages.py         REPLACE: stage 1 tests on the new model
  test_gui_app.py            REPLACE: listing assertions for the larger registry
  test_gui_data_stages.py    Normalize, Align, working matrix
  test_gui_opls_stocsy.py    OPLS-DA, STOCSY, group options
  test_gui_volcano.py        Volcano, registry order, HTTP staleness
GUI/renderer/
  api.ts                     MODIFY: StageInfo.section/ppm_views/has_table, FieldSchema.description/pick_ppm
  components/Rail.tsx        MODIFY: section headers
  components/Rail.test.tsx   NEW
  components/ParamPanel.tsx  MODIFY: hints, "Use picked ppm" button
  components/ParamPanel.test.tsx MODIFY: two new tests
  components/PlotView.tsx    MODIFY: onPick click capture
  App.tsx                    MODIFY: picked state, has_table tabs, view labels
  theme.css                  MODIFY: section, hint, pick styles
GUI/e2e/smoke.spec.ts        REPLACE: positive CSV + Phase A path
```

---

### Task 1: Stages package, dependency-graph staleness, schema options

**Files:**
- Delete: `GUI/server/metbit_server/stages.py`
- Create: `GUI/server/metbit_server/stages/__init__.py`, `stages/base.py`, `stages/data.py`, `stages/analysis.py`
- Modify (replace): `GUI/server/metbit_server/session.py`, `GUI/server/metbit_server/app.py`
- Modify (replace): `tests/gui/conftest.py`, `tests/gui/test_gui_stages.py`, `tests/gui/test_gui_app.py`

**Interfaces:**
- Consumes: stage 1 `jobs.start_job`, `jobs.Busy`.
- Produces:
  - `Session(stage_ids: list[str], after: dict[str, list[str]] | None = None)`; `Session.dependents(sid) -> list[str]` (transitive); `complete()` marks `done` dependents `stale`.
  - `stages.base.Stage` fields: `id, title, params, run, section="Analysis", table=None, figures={}, requires=None, after=[], ppm_views={}` where `ppm_views: dict[str, "x" | "customdata"]`.
  - `stages.base.Dataset(X, meta, labels, before=None)`; `X` has float ppm column labels, `labels` is a `pd.Series[str]` of group names.
  - `stages.base.DATA_ORDER = ["import", "normalize", "align"]`.
  - `working(session, upto: str | None = None) -> Dataset` (raises `ValueError("Import data first")`).
  - `groups(session) -> pd.Series`; `two_groups(session, a, b) -> (X, y)` (raises `ValueError` "Choose two different groups" / "Unknown group(s): ...").
  - `spectra_figure(X, labels, title) -> go.Figure` (decimates to at most 8000 points per trace).
  - `resolve_schema(stage, session) -> dict`: fills `enum` and `default` for properties tagged `options: "groups" | "columns"` (`pick` = default index).
  - HTTP `GET /stages` items gain `section`, `ppm_views`, `has_table`; `schema` is resolved. `GET /stages/{id}/table` returns 404 when the stage has no table.
  - Test fixture `run_all(*runs: tuple[str, dict]) -> Session` (always imports `csv_path` first).

- [ ] **Step 1: Replace the test fixtures**

Replace `tests/gui/conftest.py` with:

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import fastapi  # noqa: F401
except ImportError:  # GUI extras not installed: skip this folder
    collect_ignore_glob = ["test_*.py"]

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "GUI" / "server"))

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(0)
    ppm = np.linspace(10, 0, 40)
    X = np.abs(rng.normal(size=(12, 40))) + 1  # positive, like real spectra
    X[6:, 5] += 3  # group B carries an extra peak
    df = pd.DataFrame(X, columns=[f"{v:.4f}" for v in ppm])
    df.insert(0, "Group", ["A"] * 6 + ["B"] * 6)
    df.insert(0, "Sample", [f"s{i}" for i in range(12)])
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def run_all(csv_path):
    """Build a session by running import and then each (stage id, params) in order."""
    from metbit_server.session import Session
    from metbit_server.stages import STAGES

    by_id = {s.id: s for s in STAGES}

    def run(*runs):
        s = Session([st.id for st in STAGES], {st.id: st.after for st in STAGES})
        for sid, params in [("import", {"path": csv_path}), *runs]:
            stage = by_id[sid]
            p = stage.params.model_validate(params)
            s.complete(sid, p.model_dump(), stage.run(p, s, lambda _pct: None))
        return s

    return run
```

- [ ] **Step 2: Replace the stage 1 tests with graph-aware versions**

Replace `tests/gui/test_gui_stages.py` with:

```python
import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from metbit_server.session import Session
from metbit_server.stages import STAGES, ImportParams, PCAParams

BY_ID = {s.id: s for s in STAGES}


def _noop(_pct):
    pass


def test_import_and_pca_registration():
    assert STAGES[0].id == "import" and STAGES[0].section == "Data"
    assert BY_ID["pca"].section == "Analysis"
    assert BY_ID["pca"].requires == "import"
    assert BY_ID["pca"].after == ["import", "normalize", "align"]
    assert BY_ID["pca"].ppm_views == {"loadings": "x"}
    assert ImportParams.model_json_schema()["properties"]["path"]["format"] == "file-path"


def test_import_splits_features_metadata_and_labels(csv_path):
    data = BY_ID["import"].run(ImportParams(path=csv_path), Session(["import"]), _noop)
    assert data.X.shape == (12, 40)
    assert data.X.columns.dtype == float
    assert list(data.meta.columns) == ["Sample", "Group"]
    assert list(data.labels) == ["A"] * 6 + ["B"] * 6
    assert list(BY_ID["import"].table(data).columns) == ["Sample", "Group"]
    assert isinstance(BY_ID["import"].figures["spectra"](data), go.Figure)


def test_import_rejects_missing_group_column(csv_path):
    with pytest.raises(ValueError, match="Group column 'Class' not found"):
        BY_ID["import"].run(ImportParams(path=csv_path, group_column="Class"), Session(["import"]), _noop)


def test_import_rejects_csv_without_ppm_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Sample,Group\ns1,A\n")
    with pytest.raises(ValueError, match="No spectral columns"):
        BY_ID["import"].run(ImportParams(path=str(bad)), Session(["import"]), _noop)


def test_pca_run_figures_and_table(run_all):
    s = run_all()
    seen = []
    result = BY_ID["pca"].run(PCAParams(n_components=3), s, seen.append)
    assert seen[-1] == 1.0
    assert result.pcs == ["PC1", "PC2"]
    for view in ("scores", "loadings", "variance"):
        assert isinstance(BY_ID["pca"].figures[view](result), go.Figure)
    assert "Explained variance" in BY_ID["pca"].table(result).columns


def test_pca_params_reject_axis_beyond_components():
    with pytest.raises(ValidationError, match="exceeds"):
        PCAParams(n_components=2, pc_y=3)
    with pytest.raises(ValidationError, match="differ"):
        PCAParams(pc_x=1, pc_y=1)


def test_session_staleness_follows_dependencies():
    after = {"normalize": ["import"], "pca": ["import", "normalize"], "opls": ["import", "normalize"]}
    s = Session(["import", "normalize", "pca", "opls"], after)
    for sid in ("import", "normalize", "pca", "opls"):
        s.complete(sid, {}, sid)
    s.complete("pca", {}, "again")
    assert s.status["opls"] == "done"  # siblings never stale each other
    s.complete("import", {}, "new")
    assert s.status == {"import": "done", "normalize": "stale", "pca": "stale", "opls": "stale"}
    assert s.dependents("import") == ["normalize", "pca", "opls"]
    with pytest.raises(KeyError):
        Session(["import"]).output("import")


def test_working_matrix_and_group_options(run_all):
    from metbit_server.stages import resolve_schema
    from metbit_server.stages.base import working

    s = run_all()
    assert working(s) is s.output("import")
    empty = Session([st.id for st in STAGES])
    with pytest.raises(ValueError, match="Import data first"):
        working(empty)
    assert resolve_schema(BY_ID["pca"], s) == PCAParams.model_json_schema()  # no tagged fields
```

Replace `tests/gui/test_gui_app.py` with:

```python
import threading

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from conftest import AUTH, TOKEN
from metbit_server.app import create_app
from metbit_server.stages import Stage


@pytest.fixture
def client():
    return TestClient(create_app(TOKEN))


def run_stage(client, sid, params):
    r = client.post(f"/stages/{sid}/run", json=params, headers=AUTH)
    assert r.status_code == 200, r.text
    body = client.get(f"/jobs/{r.json()['job_id']}/events", headers=AUTH).text
    return [ln.split(": ", 1)[1] for ln in body.splitlines() if ln.startswith("event: ")]


def statuses(client):
    return {s["id"]: s["status"] for s in client.get("/stages", headers=AUTH).json()}


def test_token_required(client):
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/health", headers=AUTH).json() == {"ok": True}


def test_cors_preflight_needs_no_token(client):
    r = client.options("/health", headers={"Origin": "null", "Access-Control-Request-Method": "GET",
                                           "Access-Control-Request-Headers": "authorization"})
    assert r.status_code == 200


def test_stages_listing(client):
    stages = {s["id"]: s for s in client.get("/stages", headers=AUTH).json()}
    assert next(iter(stages)) == "import"
    assert stages["import"]["status"] == "empty"
    assert stages["import"]["section"] == "Data"
    assert stages["import"]["has_table"] is True
    assert stages["import"]["schema"]["properties"]["path"]["format"] == "file-path"
    assert stages["pca"]["views"] == ["scores", "loadings", "variance"]
    assert stages["pca"]["ppm_views"] == {"loadings": "x"}


def test_import_then_pca_then_reimport_marks_stale(client, csv_path):
    assert run_stage(client, "import", {"path": csv_path})[-1] == "done"
    assert run_stage(client, "pca", {"n_components": 2})[-1] == "done"
    assert (statuses(client)["import"], statuses(client)["pca"]) == ("done", "done")
    run_stage(client, "import", {"path": csv_path})
    assert (statuses(client)["import"], statuses(client)["pca"]) == ("done", "stale")
    assert client.get("/stages", headers=AUTH).json()[0]["params"]["path"] == csv_path


def test_pca_before_import_is_409(client):
    r = client.post("/stages/pca/run", json={}, headers=AUTH)
    assert r.status_code == 409
    assert "import" in r.json()["detail"]


def test_invalid_params_are_422_with_field_loc(client):
    r = client.post("/stages/import/run", json={"path": "x", "group_column": 5}, headers=AUTH)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["group_column"]


def test_unknown_stage_is_404(client):
    assert client.post("/stages/nope/run", json={}, headers=AUTH).status_code == 404


def test_failed_job_reports_error_and_status(client, tmp_path):
    events = run_stage(client, "import", {"path": str(tmp_path / "missing.csv")})
    assert events[-1] == "error"
    assert statuses(client)["import"] == "failed"


def test_second_run_while_busy_is_409():
    gate = threading.Event()

    class P(BaseModel):
        pass

    slow = Stage(id="slow", title="Slow", params=P, run=lambda p, s, prog: gate.wait(5), table=lambda o: None)
    client = TestClient(create_app(TOKEN, stages=[slow]))
    first = client.post("/stages/slow/run", json={}, headers=AUTH)
    assert first.status_code == 200
    assert client.post("/stages/slow/run", json={}, headers=AUTH).status_code == 409
    gate.set()
    body = client.get(f"/jobs/{first.json()['job_id']}/events", headers=AUTH).text
    assert "event: done" in body
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui --no-cov -q`
Expected: FAIL (for example `AttributeError: 'Stage' object has no attribute 'section'`, `TypeError` on `Session(..., after)`).

- [ ] **Step 4: Graph-aware session**

Replace `GUI/server/metbit_server/session.py` with:

```python
"""In-memory pipeline state for one GUI session."""
import threading
from typing import Any, Optional


class Session:
    def __init__(self, stage_ids: list[str], after: Optional[dict[str, list[str]]] = None) -> None:
        self.order = list(stage_ids)
        self.after = after or {}  # stage id -> ids whose re-run makes it stale
        self.status: dict[str, str] = {sid: "empty" for sid in self.order}
        self.params: dict[str, dict] = {}
        self.outputs: dict[str, Any] = {}
        self.lock = threading.Lock()  # one job at a time

    def dependents(self, sid: str) -> list[str]:
        """Every stage that depends on `sid`, directly or transitively."""
        found: list[str] = []
        todo = [sid]
        while todo:
            cur = todo.pop()
            for other in self.order:
                if cur in self.after.get(other, ()) and other not in found:
                    found.append(other)
                    todo.append(other)
        return found

    def start(self, sid: str) -> None:
        self.status[sid] = "running"

    def complete(self, sid: str, params: dict, output: Any) -> None:
        self.params[sid] = params
        self.outputs[sid] = output
        self.status[sid] = "done"
        for dep in self.dependents(sid):
            if self.status[dep] == "done":
                self.status[dep] = "stale"

    def fail(self, sid: str) -> None:
        self.status[sid] = "failed"

    def output(self, sid: str) -> Any:
        if sid not in self.outputs:
            raise KeyError(sid)
        return self.outputs[sid]
```

- [ ] **Step 5: Create the stages package**

Run: `git rm -q GUI/server/metbit_server/stages.py && mkdir -p GUI/server/metbit_server/stages`

Create `GUI/server/metbit_server/stages/base.py`:

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

DATA_ORDER = ["import", "normalize", "align"]
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


@dataclass
class Dataset:
    X: pd.DataFrame                        # samples x features, float ppm column labels
    meta: pd.DataFrame                     # every non-ppm column
    labels: pd.Series                      # group name per sample (str)
    before: Optional[pd.DataFrame] = None  # input matrix, for before / after views


def working(session: Session, upto: Optional[str] = None) -> Dataset:
    """Newest `done` data stage, considering only stages earlier than `upto`."""
    order = DATA_ORDER if upto is None else DATA_ORDER[: DATA_ORDER.index(upto)]
    for sid in reversed(order):
        if session.status.get(sid) == "done":
            return session.outputs[sid]
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


def option_values(session: Session, kind: str) -> Optional[list[str]]:
    if session.status.get("import") != "done":
        return None
    if kind == "groups":
        return sorted(set(groups(session)))
    if kind == "columns":
        return [str(c) for c in session.output("import").meta.columns]
    return None


def resolve_schema(stage: Stage, session: Session) -> dict:
    """Param JSON schema with data-driven `enum` and `default` for tagged fields."""
    schema = stage.params.model_json_schema()
    for prop in schema.get("properties", {}).values():
        values = option_values(session, prop.get("options", ""))
        if values:
            prop["enum"] = values
            if prop.get("default") not in values:
                prop["default"] = values[min(prop.get("pick", 0), len(values) - 1)]
    return schema
```

Create `GUI/server/metbit_server/stages/data.py`:

```python
"""Data stages: Import (CSV)."""
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from ..session import Session
from .base import Dataset, Progress, Stage, spectra_figure


class ImportParams(BaseModel):
    path: str = Field(title="CSV file", json_schema_extra={"format": "file-path"})
    group_column: str = Field("Group", title="Group column")


def _is_ppm(name: Any) -> bool:
    try:
        float(name)
        return True
    except (TypeError, ValueError):
        return False


def run_import(p: ImportParams, session: Session, progress: Progress) -> Dataset:
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
    progress(1.0)
    return Dataset(X=X, meta=meta, labels=meta[p.group_column].astype(str).reset_index(drop=True))


DATA_STAGES: list[Stage] = [
    Stage(
        id="import",
        title="Import",
        section="Data",
        params=ImportParams,
        run=run_import,
        table=lambda d: d.meta,
        figures={"spectra": lambda d: spectra_figure(d.X, d.labels, "Spectra")},
        ppm_views={"spectra": "x"},
    ),
]
```

Create `GUI/server/metbit_server/stages/analysis.py`:

```python
"""Analysis stages: PCA."""
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from metbit import pca

from ..session import Session
from .base import DATA_ORDER, Progress, Stage, groups, working

Scaling = Literal["pareto", "uv", "mean"]


class PCAParams(BaseModel):
    n_components: int = Field(2, ge=2, le=10, title="Components")
    scaling: Scaling = Field("pareto", title="Scaling")
    pc_x: int = Field(1, ge=1, title="X axis PC")
    pc_y: int = Field(2, ge=1, title="Y axis PC")

    @model_validator(mode="after")
    def _axes_valid(self) -> "PCAParams":
        if max(self.pc_x, self.pc_y) > self.n_components:
            raise ValueError("PC axis exceeds the number of components")
        if self.pc_x == self.pc_y:
            raise ValueError("X and Y axis PCs must differ")
        return self


@dataclass
class PCAResult:
    model: Any
    pcs: list[str]


def run_pca(p: PCAParams, session: Session, progress: Progress) -> PCAResult:
    X = working(session).X
    model = pca(
        X=X,
        label=groups(session),
        features_name=list(X.columns),
        n_components=p.n_components,
        scaling_method=p.scaling,
    )
    model.fit()
    progress(1.0)
    return PCAResult(model=model, pcs=[f"PC{p.pc_x}", f"PC{p.pc_y}"])


ANALYSIS_STAGES: list[Stage] = [
    Stage(
        id="pca",
        title="PCA",
        params=PCAParams,
        run=run_pca,
        table=lambda r: r.model.get_explained_variance(),
        figures={
            "scores": lambda r: r.model.plot_pca_scores(pc=r.pcs),
            "loadings": lambda r: r.model.plot_loading_(pc=r.pcs),
            "variance": lambda r: r.model.plot_observe_variance(),
        },
        requires="import",
        after=list(DATA_ORDER),
        ppm_views={"loadings": "x"},
    ),
]
```

Create `GUI/server/metbit_server/stages/__init__.py`:

```python
"""Stage registry, in rail order."""
from .analysis import ANALYSIS_STAGES, PCAParams, PCAResult
from .base import DATA_ORDER, Dataset, Progress, Stage, resolve_schema
from .data import DATA_STAGES, ImportParams

STAGES: list[Stage] = [*DATA_STAGES, *ANALYSIS_STAGES]
```

- [ ] **Step 6: Update the app**

Replace `GUI/server/metbit_server/app.py` with:

```python
"""FastAPI sidecar for the metbit desktop GUI."""
import asyncio
import json
import secrets

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from .jobs import Busy, Job, start_job
from .session import Session
from .stages import STAGES, Stage, resolve_schema


def create_app(token: str, stages: list[Stage] = STAGES) -> FastAPI:
    by_id = {s.id: s for s in stages}
    session = Session([s.id for s in stages], {s.id: s.after for s in stages})
    jobs: dict[str, Job] = {}

    def check_token(request: Request) -> None:
        got = request.headers.get("authorization", "")
        if not secrets.compare_digest(got, f"Bearer {token}"):
            raise HTTPException(401, "Invalid token")

    app = FastAPI(dependencies=[Depends(check_token)])
    # ponytail: "*" is safe because every route needs the per-session bearer token;
    # CORS answers preflight before routing, so OPTIONS needs no token.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def stage_or_404(sid: str) -> Stage:
        if sid not in by_id:
            raise HTTPException(404, f"Unknown stage '{sid}'")
        return by_id[sid]

    def result_or_409(sid: str):
        stage_or_404(sid)
        if session.status[sid] not in ("done", "stale"):
            raise HTTPException(409, f"Stage '{sid}' has no result")
        return session.outputs[sid]

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/stages")
    def list_stages():
        return [
            {
                "id": s.id,
                "title": s.title,
                "section": s.section,
                "requires": s.requires,
                "status": session.status[s.id],
                "views": list(s.figures),
                "ppm_views": s.ppm_views,
                "has_table": s.table is not None,
                "schema": resolve_schema(s, session),
                "params": session.params.get(s.id),
            }
            for s in stages
        ]

    @app.post("/stages/{sid}/run")
    def run(sid: str, body: dict = Body(default={})):
        stage = stage_or_404(sid)
        try:
            params = stage.params.model_validate(body)
        except ValidationError as e:
            raise HTTPException(422, json.loads(e.json(include_url=False)))
        if stage.requires and session.status[stage.requires] != "done":
            raise HTTPException(409, f"Run '{stage.requires}' first")
        try:
            job = start_job(session, stage, params)
        except Busy:
            raise HTTPException(409, "Another job is running")
        jobs[job.id] = job
        return {"job_id": job.id}

    @app.get("/jobs/{job_id}/events")
    async def events(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")

        async def stream():
            while True:
                kind, data = await asyncio.to_thread(job.events.get)
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
                if kind in ("done", "error"):
                    jobs.pop(job_id, None)
                    return

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/stages/{sid}/figure")
    def figure(sid: str, view: str):
        out = result_or_409(sid)
        make = by_id[sid].figures.get(view)
        if make is None:
            raise HTTPException(404, f"Unknown view '{view}'")
        fig = make(out)
        fig.update_layout(width=None, height=None, autosize=True)  # renderer sizes the plot
        return Response(fig.to_json(), media_type="application/json")

    @app.get("/stages/{sid}/table")
    def table(sid: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
        if stage_or_404(sid).table is None:
            raise HTTPException(404, f"Stage '{sid}' has no table")
        df = by_id[sid].table(result_or_409(sid))
        page = json.loads(df.iloc[offset:offset + limit].to_json(orient="split", index=False))
        return {"columns": [str(c) for c in page["columns"]], "rows": page["data"], "total": len(df)}

    return app
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q`
Expected: `21 passed` (8 stage tests + 9 app tests + 4 view tests).

- [ ] **Step 8: Commit**

```bash
git add -A GUI/server tests/gui
git commit -m "refactor(gui): split stages into a package with dependency-graph staleness and schema options"
```

---

### Task 2: Normalize and Align stages

**Files:**
- Modify (replace): `GUI/server/metbit_server/stages/data.py`, `GUI/server/metbit_server/stages/__init__.py`
- Test: `tests/gui/test_gui_data_stages.py`

**Interfaces:**
- Consumes: `Stage`, `Dataset`, `working`, `spectra_figure`, `DATA_ORDER` from Task 1.
- Produces:
  - `NormalizeParams(method: "pqn" | "msc" | "snv" | "snv_pqn" = "pqn")`; stage id `normalize`, views `spectra`, `before`, `after=["import"]`.
  - `AlignParams(windows: str = "", reference: "median" | "mean" = "median", max_shift_ppm: float = 0.02)`; stage id `align`, views `spectra`, `before`, `after=["import", "normalize"]`.
  - `parse_windows(text: str) -> list[tuple[float, float]]` returning `(high, low)` pairs; raises `ValueError("Cannot read window '<part>': use start-end, for example 3.2-2.8")`.
  - Both outputs are `Dataset` with `before` = the input matrix object.

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_data_stages.py`:

```python
import plotly.graph_objects as go
import pytest

from metbit_server.stages import STAGES, parse_windows
from metbit_server.stages.base import working

BY_ID = {s.id: s for s in STAGES}


def test_data_stage_registration():
    assert [s.id for s in STAGES][:3] == ["import", "normalize", "align"]
    assert BY_ID["normalize"].after == ["import"]
    assert BY_ID["align"].after == ["import", "normalize"]
    assert BY_ID["align"].ppm_views == {"spectra": "x", "before": "x"}


@pytest.mark.parametrize("method", ["pqn", "msc", "snv", "snv_pqn"])
def test_normalize_uses_import_and_becomes_working_matrix(run_all, method):
    s = run_all(("normalize", {"method": method}))
    out = s.output("normalize")
    assert out.X.shape == (12, 40)
    assert out.before is s.output("import").X
    assert not out.X.equals(out.before)
    assert list(out.X.columns) == list(out.before.columns)
    assert working(s) is out
    for view in ("spectra", "before"):
        assert isinstance(BY_ID["normalize"].figures[view](out), go.Figure)


def test_align_reads_normalized_matrix_and_defaults_to_whole_spectrum(run_all):
    s = run_all(("normalize", {"method": "snv"}), ("align", {}))
    out = s.output("align")
    assert out.before is s.output("normalize").X
    assert list(out.X.columns) == list(out.before.columns)
    assert working(s) is out


def test_align_with_explicit_windows(run_all):
    s = run_all(("align", {"windows": "9-7, 3-1", "reference": "mean", "max_shift_ppm": 0.1}))
    assert s.output("align").X.shape == (12, 40)


def test_parse_windows():
    assert parse_windows("") == []
    assert parse_windows("3.2-2.8, 1.2 - 1.5") == [(3.2, 2.8), (1.5, 1.2)]
    assert parse_windows("0.5--0.2") == [(0.5, -0.2)]
    with pytest.raises(ValueError, match="Cannot read window 'abc'"):
        parse_windows("abc")


def test_bad_windows_are_a_validation_error():
    from pydantic import ValidationError

    from metbit_server.stages import AlignParams

    with pytest.raises(ValidationError, match="Cannot read window"):
        AlignParams(windows="3.2 to nowhere")


def test_stale_normalize_is_skipped_by_working(run_all, csv_path):
    s = run_all(("normalize", {"method": "pqn"}), ("import", {"path": csv_path}))
    assert s.status["normalize"] == "stale"
    assert working(s) is s.output("import")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_data_stages.py --no-cov -q`
Expected: FAIL with `ImportError: cannot import name 'parse_windows'`.

- [ ] **Step 3: Implement the data stages**

Replace `GUI/server/metbit_server/stages/data.py` with:

```python
"""Data stages: Import (CSV), Normalize, Align."""
import re
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from metbit.nmr.alignment import icoshift_align
from metbit.preprocessing.normalize import Normalization

from ..session import Session
from .base import Dataset, Progress, Stage, spectra_figure, working


# Import (CSV) ---------------------------------------------------------------

class ImportParams(BaseModel):
    path: str = Field(title="CSV file", json_schema_extra={"format": "file-path"})
    group_column: str = Field("Group", title="Group column")


def _is_ppm(name: Any) -> bool:
    try:
        float(name)
        return True
    except (TypeError, ValueError):
        return False


def run_import(p: ImportParams, session: Session, progress: Progress) -> Dataset:
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
    progress(1.0)
    return Dataset(X=X, meta=meta, labels=meta[p.group_column].astype(str).reset_index(drop=True))


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
        figures={"spectra": lambda d: spectra_figure(d.X, d.labels, "Spectra")},
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
from .analysis import ANALYSIS_STAGES, PCAParams, PCAResult
from .base import DATA_ORDER, Dataset, Progress, Stage, resolve_schema
from .data import DATA_STAGES, AlignParams, ImportParams, NormalizeParams, parse_windows

STAGES: list[Stage] = [*DATA_STAGES, *ANALYSIS_STAGES]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q`
Expected: `31 passed`.

- [ ] **Step 5: Commit**

```bash
git add GUI/server tests/gui
git commit -m "feat(gui): add Normalize and Align data stages"
```

---

### Task 3: OPLS-DA and STOCSY stages

**Files:**
- Modify (replace): `GUI/server/metbit_server/stages/analysis.py`, `GUI/server/metbit_server/stages/__init__.py`
- Test: `tests/gui/test_gui_opls_stocsy.py`

**Interfaces:**
- Consumes: `two_groups`, `working`, `groups`, `resolve_schema`, `DATA_ORDER` from Task 1.
- Produces:
  - `OPLSParams(group_a: str = "", group_b: str = "", scaling="pareto", kfold: int = 3, vip_threshold: float = 1.0, permutations: int = 0)`; `group_a`/`group_b` tagged `options: "groups"` with `pick` 0 / 1.
  - `OPLSResult(model, vip_threshold: float, permuted: bool)`; stage id `opls`, title `OPLS-DA`, views `scores`, `s-plot`, `vip`, `permutation`; table = VIP scores sorted descending; `ppm_views={"s-plot": "customdata", "vip": "x"}`.
  - `STOCSYParams(anchor_ppm: float (required, tagged pick_ppm), p_threshold: float = 1e-4)`; stage id `stocsy`, view `correlation`, no table, `ppm_views={"correlation": "x"}`; raises `ValueError("Anchor <v> ppm is outside the spectrum (<lo> to <hi> ppm)")`.
  - `not_run_figure(message: str) -> go.Figure` (blank figure with one centered annotation).

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_opls_stocsy.py`:

```python
import plotly.graph_objects as go
import pytest

from metbit_server.session import Session
from metbit_server.stages import STAGES, OPLSParams, resolve_schema

BY_ID = {s.id: s for s in STAGES}


def test_group_options_are_injected_after_import(run_all):
    fresh = Session([st.id for st in STAGES])
    assert "enum" not in resolve_schema(BY_ID["opls"], fresh)["properties"]["group_a"]
    props = resolve_schema(BY_ID["opls"], run_all())["properties"]
    assert props["group_a"]["enum"] == ["A", "B"]
    assert (props["group_a"]["default"], props["group_b"]["default"]) == ("A", "B")


def test_opls_views_table_and_default_permutation_placeholder(run_all):
    r = run_all(("opls", {"group_a": "A", "group_b": "B"})).output("opls")
    for view in ("scores", "s-plot", "vip", "permutation"):
        assert isinstance(BY_ID["opls"].figures[view](r), go.Figure)
    assert "not run" in BY_ID["opls"].figures["permutation"](r).layout.annotations[0].text
    vip = BY_ID["opls"].table(r)
    assert list(vip["VIP"]) == sorted(vip["VIP"], reverse=True)
    assert BY_ID["opls"].ppm_views == {"s-plot": "customdata", "vip": "x"}


def test_opls_permutation_test(run_all):
    r = run_all(("opls", {"group_a": "A", "group_b": "B", "permutations": 5})).output("opls")
    assert r.permuted
    assert isinstance(BY_ID["opls"].figures["permutation"](r), go.Figure)


@pytest.mark.parametrize("a,b,msg", [("A", "A", "two different"), ("A", "Z", "Unknown group")])
def test_two_group_validation(run_all, a, b, msg):
    with pytest.raises(ValueError, match=msg):
        run_all(("opls", {"group_a": a, "group_b": b}))


def test_opls_params_bounds():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        OPLSParams(kfold=1)
    with pytest.raises(ValidationError):
        OPLSParams(permutations=-1)


def test_stocsy_runs_and_checks_anchor_range(run_all):
    fig = run_all(("stocsy", {"anchor_ppm": 5.0})).output("stocsy")
    assert isinstance(BY_ID["stocsy"].figures["correlation"](fig), go.Figure)
    assert BY_ID["stocsy"].table is None
    assert BY_ID["stocsy"].params.model_json_schema()["properties"]["anchor_ppm"]["pick_ppm"] is True
    with pytest.raises(ValueError, match="outside the spectrum"):
        run_all(("stocsy", {"anchor_ppm": 50.0}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_opls_stocsy.py --no-cov -q`
Expected: FAIL with `ImportError: cannot import name 'OPLSParams'`.

- [ ] **Step 3: Implement the stages**

Replace `GUI/server/metbit_server/stages/analysis.py` with:

```python
"""Analysis stages: PCA, OPLS-DA, STOCSY."""
from dataclasses import dataclass
from typing import Any, Literal

import plotly.graph_objects as go
from pydantic import BaseModel, Field, model_validator

from metbit import opls_da, pca
from metbit.analysis.stocsy import STOCSY

from ..session import Session
from .base import DATA_ORDER, Progress, Stage, groups, two_groups, working

Scaling = Literal["pareto", "uv", "mean"]


def not_run_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# PCA ------------------------------------------------------------------------

class PCAParams(BaseModel):
    n_components: int = Field(2, ge=2, le=10, title="Components")
    scaling: Scaling = Field("pareto", title="Scaling")
    pc_x: int = Field(1, ge=1, title="X axis PC")
    pc_y: int = Field(2, ge=1, title="Y axis PC")

    @model_validator(mode="after")
    def _axes_valid(self) -> "PCAParams":
        if max(self.pc_x, self.pc_y) > self.n_components:
            raise ValueError("PC axis exceeds the number of components")
        if self.pc_x == self.pc_y:
            raise ValueError("X and Y axis PCs must differ")
        return self


@dataclass
class PCAResult:
    model: Any
    pcs: list[str]


def run_pca(p: PCAParams, session: Session, progress: Progress) -> PCAResult:
    X = working(session).X
    model = pca(
        X=X,
        label=groups(session),
        features_name=list(X.columns),
        n_components=p.n_components,
        scaling_method=p.scaling,
    )
    model.fit()
    progress(1.0)
    return PCAResult(model=model, pcs=[f"PC{p.pc_x}", f"PC{p.pc_y}"])


# OPLS-DA --------------------------------------------------------------------

def _group_field(title: str, pick: int):
    return Field("", title=title, json_schema_extra={"options": "groups", "pick": pick})


class OPLSParams(BaseModel):
    group_a: str = _group_field("Group A", 0)
    group_b: str = _group_field("Group B", 1)
    scaling: Scaling = Field("pareto", title="Scaling")
    kfold: int = Field(3, ge=2, le=10, title="CV folds")
    vip_threshold: float = Field(1.0, gt=0, title="VIP threshold")
    permutations: int = Field(0, ge=0, le=1000, title="Permutations", description="0 skips the permutation test")


@dataclass
class OPLSResult:
    model: Any
    vip_threshold: float
    permuted: bool


def run_opls(p: OPLSParams, session: Session, progress: Progress) -> OPLSResult:
    X, y = two_groups(session, p.group_a, p.group_b)
    model = opls_da(X=X, y=y, features_name=list(X.columns), scaling_method=p.scaling, kfold=p.kfold)
    model.fit()
    model.vip_scores()
    if p.permutations:
        progress(0.3)
        model.permutation_test(n_permutations=p.permutations, cv=p.kfold, n_jobs=-1, verbose=0)
    progress(1.0)
    return OPLSResult(model=model, vip_threshold=p.vip_threshold, permuted=p.permutations > 0)


def _permutation_view(r: OPLSResult) -> go.Figure:
    if not r.permuted:
        return not_run_figure("Permutation test not run. Set Permutations above 0 and run again.")
    return r.model.plot_hist()


# STOCSY ---------------------------------------------------------------------

class STOCSYParams(BaseModel):
    anchor_ppm: float = Field(title="Anchor (ppm)", json_schema_extra={"pick_ppm": True})
    p_threshold: float = Field(1e-4, gt=0, lt=1, title="p threshold")


def run_stocsy(p: STOCSYParams, session: Session, progress: Progress) -> go.Figure:
    X = working(session).X
    lo, hi = float(X.columns.min()), float(X.columns.max())
    if not lo <= p.anchor_ppm <= hi:
        raise ValueError(f"Anchor {p.anchor_ppm} ppm is outside the spectrum ({lo:.2f} to {hi:.2f} ppm)")
    fig = STOCSY(X, p.anchor_ppm, p.p_threshold)
    progress(1.0)
    return fig


ANALYSIS_STAGES: list[Stage] = [
    Stage(
        id="pca",
        title="PCA",
        params=PCAParams,
        run=run_pca,
        table=lambda r: r.model.get_explained_variance(),
        figures={
            "scores": lambda r: r.model.plot_pca_scores(pc=r.pcs),
            "loadings": lambda r: r.model.plot_loading_(pc=r.pcs),
            "variance": lambda r: r.model.plot_observe_variance(),
        },
        requires="import",
        after=list(DATA_ORDER),
        ppm_views={"loadings": "x"},
    ),
    Stage(
        id="opls",
        title="OPLS-DA",
        params=OPLSParams,
        run=run_opls,
        table=lambda r: r.model.get_vip_scores().sort_values("VIP", ascending=False),
        figures={
            "scores": lambda r: r.model.plot_oplsda_scores(),
            "s-plot": lambda r: r.model.plot_s_scores(),
            "vip": lambda r: r.model.vip_plot(threshold=r.vip_threshold),
            "permutation": _permutation_view,
        },
        requires="import",
        after=list(DATA_ORDER),
        ppm_views={"s-plot": "customdata", "vip": "x"},
    ),
    Stage(
        id="stocsy",
        title="STOCSY",
        params=STOCSYParams,
        run=run_stocsy,
        figures={"correlation": lambda fig: fig},
        requires="import",
        after=list(DATA_ORDER),
        ppm_views={"correlation": "x"},
    ),
]
```

Replace `GUI/server/metbit_server/stages/__init__.py` with:

```python
"""Stage registry, in rail order."""
from .analysis import ANALYSIS_STAGES, OPLSParams, OPLSResult, PCAParams, PCAResult, STOCSYParams
from .base import DATA_ORDER, Dataset, Progress, Stage, resolve_schema
from .data import DATA_STAGES, AlignParams, ImportParams, NormalizeParams, parse_windows

STAGES: list[Stage] = [*DATA_STAGES, *ANALYSIS_STAGES]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q`
Expected: `38 passed`.

- [ ] **Step 5: Commit**

```bash
git add GUI/server tests/gui
git commit -m "feat(gui): add OPLS-DA and STOCSY stages"
```

---

### Task 4: Volcano stage and pipeline-level HTTP checks

**Files:**
- Create: `GUI/server/metbit_server/stages/stats.py`
- Modify (replace): `GUI/server/metbit_server/stages/__init__.py`
- Test: `tests/gui/test_gui_volcano.py`

**Interfaces:**
- Consumes: `two_groups` from Task 1; `OPLSParams`-style group fields.
- Produces:
  - `VolcanoParams(group_a="", group_b="", fc_threshold: float = 1.0, p_threshold: float = 0.05, correction: "fdr_bh" | "bonferroni" | "none" = "fdr_bh")`.
  - Stage id `volcano`, title `Volcano`, section `Statistics`, view `volcano`, table = `VolcanoPlot.get_table()` (columns include `feature`, `log2FC`, `p_value`, `p_adj`, `label`).
  - Final `STAGES` order: `import, normalize, align, pca, opls, stocsy, volcano`.

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_volcano.py`:

```python
import plotly.graph_objects as go
from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app
from metbit_server.stages import STAGES

BY_ID = {s.id: s for s in STAGES}


def test_registry_order_and_sections():
    assert [s.id for s in STAGES] == ["import", "normalize", "align", "pca", "opls", "stocsy", "volcano"]
    assert [s.section for s in STAGES] == ["Data"] * 3 + ["Analysis"] * 3 + ["Statistics"]


def test_volcano_table_and_figure(run_all):
    out = run_all(("volcano", {"group_a": "A", "group_b": "B"})).output("volcano")
    table = BY_ID["volcano"].table(out)
    assert {"feature", "log2FC", "p_value", "p_adj", "label"} <= set(table.columns)
    assert len(table) == 40
    assert isinstance(BY_ID["volcano"].figures["volcano"](out), go.Figure)


def test_volcano_without_correction(run_all):
    out = run_all(("volcano", {"group_a": "B", "group_b": "A", "correction": "none"})).output("volcano")
    assert "label" in BY_ID["volcano"].table(out).columns


def _run(c, sid, params):
    job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
    assert "event: done" in c.get(f"/jobs/{job}/events", headers=AUTH).text


def _statuses(c):
    return {s["id"]: s["status"] for s in c.get("/stages", headers=AUTH).json()}


def test_http_reimport_stales_analyses_but_siblings_stay(csv_path):
    c = TestClient(create_app(TOKEN))
    _run(c, "import", {"path": csv_path})
    _run(c, "pca", {})
    _run(c, "volcano", {"group_a": "A", "group_b": "B"})
    _run(c, "pca", {})
    st = _statuses(c)
    assert (st["pca"], st["volcano"]) == ("done", "done")
    _run(c, "normalize", {"method": "pqn"})
    st = _statuses(c)
    assert (st["import"], st["normalize"], st["pca"], st["volcano"]) == ("done", "done", "stale", "stale")


def test_http_listing_injects_groups_and_flags_tables(csv_path):
    c = TestClient(create_app(TOKEN))
    assert c.get("/stages/stocsy/table", headers=AUTH).status_code == 404
    _run(c, "import", {"path": csv_path})
    stages = {s["id"]: s for s in c.get("/stages", headers=AUTH).json()}
    assert stages["volcano"]["schema"]["properties"]["group_b"]["enum"] == ["A", "B"]
    assert stages["volcano"]["schema"]["properties"]["group_b"]["default"] == "B"
    assert stages["stocsy"]["has_table"] is False
    assert stages["volcano"]["section"] == "Statistics"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_volcano.py --no-cov -q`
Expected: FAIL (`KeyError: 'volcano'` and the registry order assertion).

- [ ] **Step 3: Implement the stage**

Create `GUI/server/metbit_server/stages/stats.py`:

```python
"""Statistics stages: Volcano."""
from typing import Literal

from pydantic import BaseModel, Field

from metbit.stats.multitest import VolcanoPlot

from ..session import Session
from .base import DATA_ORDER, Progress, Stage, two_groups


def _group_field(title: str, pick: int):
    return Field("", title=title, json_schema_extra={"options": "groups", "pick": pick})


class VolcanoParams(BaseModel):
    group_a: str = _group_field("Group A (reference)", 0)
    group_b: str = _group_field("Group B", 1)
    fc_threshold: float = Field(1.0, ge=0, title="log2 fold-change threshold")
    p_threshold: float = Field(0.05, gt=0, lt=1, title="p threshold")
    correction: Literal["fdr_bh", "bonferroni", "none"] = Field("fdr_bh", title="Multiple testing")


def run_volcano(p: VolcanoParams, session: Session, progress: Progress) -> VolcanoPlot:
    X, y = two_groups(session, p.group_a, p.group_b)
    df = X.copy()
    df.columns = [str(c) for c in X.columns]  # full precision: 4 dp can collide at 0.0003 ppm spacing
    value_cols = list(df.columns)
    df.insert(0, "Group", y.to_numpy())
    result = VolcanoPlot(
        df,
        group_col="Group",
        value_cols=value_cols,
        group_a=p.group_a,
        group_b=p.group_b,
        p_value_threshold=p.p_threshold,
        fc_threshold=p.fc_threshold,
        correct_p=None if p.correction == "none" else p.correction,
    )
    progress(1.0)
    return result


STATS_STAGES: list[Stage] = [
    Stage(
        id="volcano",
        title="Volcano",
        section="Statistics",
        params=VolcanoParams,
        run=run_volcano,
        table=lambda v: v.get_table(),
        figures={"volcano": lambda v: v.plot()},
        requires="import",
        after=list(DATA_ORDER),
    ),
]
```

Replace `GUI/server/metbit_server/stages/__init__.py` with:

```python
"""Stage registry, in rail order."""
from .analysis import ANALYSIS_STAGES, OPLSParams, OPLSResult, PCAParams, PCAResult, STOCSYParams
from .base import DATA_ORDER, Dataset, Progress, Stage, resolve_schema
from .data import DATA_STAGES, AlignParams, ImportParams, NormalizeParams, parse_windows
from .stats import STATS_STAGES, VolcanoParams

STAGES: list[Stage] = [*DATA_STAGES, *ANALYSIS_STAGES, *STATS_STAGES]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q`
Expected: `43 passed`.

- [ ] **Step 5: Commit**

```bash
git add GUI/server tests/gui
git commit -m "feat(gui): add Volcano statistics stage"
```

---

### Task 5: Renderer: rail sections, field hints, picked-ppm button

**Files:**
- Modify (replace): `GUI/renderer/api.ts` (type additions only), `GUI/renderer/components/Rail.tsx`, `GUI/renderer/components/ParamPanel.tsx`
- Create: `GUI/renderer/components/Rail.test.tsx`
- Modify (replace): `GUI/renderer/components/ParamPanel.test.tsx`

**Interfaces:**
- Consumes: `/stages` fields `section`, `ppm_views`, `has_table`; schema keys `description`, `pick_ppm`.
- Produces:
  - `api.ts`: `FieldSchema` gains `description?: string; pick_ppm?: boolean`; `StageInfo` gains `section: string; ppm_views: Record<string, "x" | "customdata">; has_table: boolean`.
  - `Rail`: renders `.rail-section` headers when the section changes.
  - `ParamPanel` gains optional prop `picked?: number | null`; pick fields show a `.pick` button with text `Use <value to 3 dp> ppm`; descriptions render as `.hint`.

- [ ] **Step 1: Write the failing tests**

Create `GUI/renderer/components/Rail.test.tsx`:

```tsx
import { render } from "preact";
import { expect, it } from "vitest";
import type { StageInfo } from "../api";
import { Rail } from "./Rail";

const stage = (id: string, section: string, requires: string | null = null, status: StageInfo["status"] = "empty"): StageInfo => ({
  id,
  title: id.toUpperCase(),
  section,
  requires,
  status,
  views: [],
  ppm_views: {},
  has_table: true,
  schema: { properties: {} },
  params: null,
});

it("renders one header per section and gates stages on their requirement", () => {
  const root = document.createElement("div");
  const stages = [stage("import", "Data"), stage("normalize", "Data", "import"), stage("pca", "Analysis", "import")];
  render(<Rail stages={stages} current="import" onSelect={() => {}} />, root);
  expect([...root.querySelectorAll(".rail-section")].map((e) => e.textContent)).toEqual(["Data", "Analysis"]);
  const button = (text: string) => [...root.querySelectorAll("button")].find((b) => b.textContent === text)!;
  expect(button("IMPORT").disabled).toBe(false);
  expect(button("PCA").disabled).toBe(true);
});
```

Replace `GUI/renderer/components/ParamPanel.test.tsx` with:

```tsx
import { render } from "preact";
import { expect, it, vi } from "vitest";
import type { JsonSchema } from "../api";
import { ParamPanel, fieldErrors } from "./ParamPanel";

const schema: JsonSchema = {
  properties: {
    path: { type: "string", title: "CSV file", format: "file-path" },
    n_components: { type: "integer", title: "Components", default: 2, minimum: 2, maximum: 10 },
    scaling: { type: "string", title: "Scaling", enum: ["pareto", "uv", "mean"], default: "pareto" },
  },
  required: ["path"],
};

function mount(props: Partial<Parameters<typeof ParamPanel>[0]> = {}) {
  const root = document.createElement("div");
  const onRun = vi.fn();
  render(
    <ParamPanel title="PCA" schema={schema} initial={null} errors={{}} busy={false} onRun={onRun} {...props} />,
    root,
  );
  return { root, onRun };
}

const tick = () => new Promise((r) => setTimeout(r));

it("renders one control per property with the right kind", () => {
  const { root } = mount();
  expect(root.querySelector<HTMLInputElement>("#f-n_components")!.type).toBe("number");
  expect(root.querySelector("#f-scaling")!.tagName).toBe("SELECT");
  expect(root.querySelector("#f-path")).not.toBeNull();
  expect(root.textContent).toContain("Browse");
  expect(root.textContent).toContain("Components");
});

it("submits defaults merged with initial values", () => {
  const { root, onRun } = mount({ initial: { path: "/tmp/a.csv" } });
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onRun).toHaveBeenCalledWith({ path: "/tmp/a.csv", n_components: 2, scaling: "pareto" });
});

it("marks fields named in errors as invalid and shows form-level errors", () => {
  const { root } = mount({ errors: { n_components: "too small", "": "axes clash" } });
  expect(root.querySelector("#f-n_components")!.closest(".field")!.classList.contains("invalid")).toBe(true);
  expect(root.textContent).toContain("too small");
  expect(root.textContent).toContain("axes clash");
});

it("disables Run while busy", () => {
  const { root } = mount({ busy: true });
  expect(root.querySelector<HTMLButtonElement>("button[type=submit]")!.disabled).toBe(true);
});

it("maps pydantic 422 detail to field errors", () => {
  expect(fieldErrors([{ loc: ["n_components"], msg: "bad" }, { loc: [], msg: "whole" }])).toEqual({
    n_components: "bad",
    "": "whole",
  });
  expect(fieldErrors("boom")).toEqual({ "": "boom" });
});

it("shows field descriptions as hints", () => {
  const withHint: JsonSchema = { properties: { windows: { type: "string", title: "Windows", description: "e.g. 3.2-2.8" } } };
  const { root } = mount({ schema: withHint });
  expect(root.querySelector(".hint")!.textContent).toBe("e.g. 3.2-2.8");
});

it("offers the picked ppm for pick fields only", async () => {
  const pickSchema: JsonSchema = {
    properties: {
      anchor_ppm: { type: "number", title: "Anchor (ppm)", pick_ppm: true },
      p_threshold: { type: "number", title: "p", default: 0.0001 },
    },
  };
  const { root, onRun } = mount({ schema: pickSchema, picked: 3.0512 });
  const picks = root.querySelectorAll<HTMLButtonElement>(".pick");
  expect(picks).toHaveLength(1);
  expect(picks[0].textContent).toBe("Use 3.051 ppm");
  picks[0].click();
  await tick();
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onRun).toHaveBeenCalledWith({ anchor_ppm: 3.0512, p_threshold: 0.0001 });
});

it("hides the pick button when nothing is picked", () => {
  const pickSchema: JsonSchema = { properties: { anchor_ppm: { type: "number", pick_ppm: true } } };
  const { root } = mount({ schema: pickSchema, picked: null });
  expect(root.querySelector(".pick")).toBeNull();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd GUI && npm test`
Expected: FAIL (no `.rail-section`, no `.hint`, no `.pick`; typecheck of `ppm_views` in the Rail test fails under `npm run typecheck`).

- [ ] **Step 3: Extend the API types**

In `GUI/renderer/api.ts`, replace the `FieldSchema` and `StageInfo` interfaces with:

```ts
export interface FieldSchema {
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  enum?: string[];
  minimum?: number;
  maximum?: number;
  format?: string;
  pick_ppm?: boolean;
}
```

```ts
export interface StageInfo {
  id: string;
  title: string;
  section: string;
  requires: string | null;
  status: Status;
  views: string[];
  ppm_views: Record<string, "x" | "customdata">;
  has_table: boolean;
  schema: JsonSchema;
  params: Record<string, unknown> | null;
}
```

- [ ] **Step 4: Rail sections**

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
            <button aria-current={s.id === current} disabled={!reachable} onClick={() => onSelect(s.id)}>
              <span class={`dot ${s.status}`} title={s.status} />
              {s.title}
            </button>
          </Fragment>
        );
      })}
    </nav>
  );
}
```

- [ ] **Step 5: Hints and pick button in the parameter panel**

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
      {Object.entries(schema.properties).map(([k, f]) => (
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
  } else if (field.format === "file-path") {
    // ponytail: CSV is the only file input so far; take filters from the schema when Bruker import lands
    const browse = async () => {
      const path = await window.metbit.openFile([{ name: "CSV", extensions: ["csv"] }]);
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

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd GUI && npm run typecheck && npm test`
Expected: typecheck exits 0; `13 passed` (2 sidecar + 2 api + 1 rail + 8 param panel).

- [ ] **Step 7: Commit**

```bash
git add GUI/renderer
git commit -m "feat(gui): add rail sections, field hints and a picked-ppm button"
```

---

### Task 6: Click-to-pick plots and App wiring

**Files:**
- Modify (replace): `GUI/renderer/components/PlotView.tsx`, `GUI/renderer/App.tsx`
- Modify: `GUI/renderer/theme.css` (append rules)

**Interfaces:**
- Consumes: `StageInfo.ppm_views`, `StageInfo.has_table`; `ParamPanel` prop `picked`.
- Produces:
  - `PlotView({ figure, onPick?: (point: PickPoint) => void })`, `PickPoint = { x: unknown; customdata: unknown }`.
  - `App` keeps `picked: number | null`, set only from views listed in the current stage's `ppm_views`; shows `.picked` text `δ <value> ppm` in the tab bar; tabs = views plus `table` only when `has_table`; tab labels map `vip` to `VIP` and `s-plot` to `S-plot`.

- [ ] **Step 1: Plot click capture**

Replace `GUI/renderer/components/PlotView.tsx` with:

```tsx
import Plotly from "plotly.js-dist-min";
import { useEffect, useRef } from "preact/hooks";

interface Figure {
  data: unknown[];
  layout: Record<string, any>;
}

export interface PickPoint {
  x: unknown;
  customdata: unknown;
}

const BLACK = new Set(["black", "#000", "#000000", "rgb(0,0,0)", "rgb(0, 0, 0)"]);

// metbit figures use plotly_white with hard-coded black lines and labels; recolor them from the CSS tokens
// ponytail: read once per render, so an OS theme switch applies on the next figure load
function themed(layout: Record<string, any>): Record<string, any> {
  const css = getComputedStyle(document.documentElement);
  const text = css.getPropertyValue("--text").trim();
  const border = css.getPropertyValue("--border").trim();
  const out: Record<string, any> = {
    ...layout,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { ...layout.font, family: css.getPropertyValue("--font"), size: 12, color: text },
  };
  for (const key of Object.keys(layout)) {
    if (/^[xy]axis\d*$/.test(key)) {
      out[key] = { ...layout[key], gridcolor: border, linecolor: text, zerolinecolor: border, tickfont: { ...layout[key].tickfont, color: text } };
    }
  }
  if (layout.title) out.title = { ...layout.title, font: { ...layout.title.font, color: text } };
  if (layout.annotations) out.annotations = layout.annotations.map((a: any) => ({ ...a, font: { ...a.font, color: text } }));
  if (layout.shapes) {
    out.shapes = layout.shapes.map((s: any) => (BLACK.has(String(s.line?.color).toLowerCase()) ? { ...s, line: { ...s.line, color: text } } : s));
  }
  return out;
}

export function PlotView({ figure, onPick }: { figure: Figure; onPick?: (point: PickPoint) => void }) {
  const el = useRef<HTMLDivElement>(null);
  const pick = useRef(onPick);
  pick.current = onPick; // latest callback without re-binding plotly listeners

  useEffect(() => {
    const node = el.current!;
    return () => Plotly.purge(node);
  }, []);

  useEffect(() => {
    const node: any = el.current!;
    Plotly.react(node, figure.data, themed(figure.layout), { responsive: true, displaylogo: false }).then(() => {
      node.removeAllListeners?.("plotly_click");
      node.on("plotly_click", (e: { points?: PickPoint[] }) => {
        const p = e.points?.[0];
        if (p) pick.current?.({ x: p.x, customdata: p.customdata });
      });
    });
  }, [figure]);

  return <div ref={el} class="plot" />;
}
```

- [ ] **Step 2: App wiring**

Replace `GUI/renderer/App.tsx` with:

```tsx
import { useEffect, useState } from "preact/hooks";
import { Api, ApiError, type StageInfo, type TablePage } from "./api";
import { DataTable } from "./components/DataTable";
import { JobBar } from "./components/JobBar";
import { ParamPanel, fieldErrors, type FieldErrors, type Values } from "./components/ParamPanel";
import { PlotView, type PickPoint } from "./components/PlotView";
import { Rail } from "./components/Rail";

interface Failure {
  message: string;
  traceback?: string;
}

const LABELS: Record<string, string> = { vip: "VIP", "s-plot": "S-plot" };
const label = (t: string) => LABELS[t] ?? t[0].toUpperCase() + t.slice(1);

export function App() {
  const [api, setApi] = useState<Api | null>(null);
  const [stages, setStages] = useState<StageInfo[]>([]);
  const [current, setCurrent] = useState("import");
  const [tab, setTab] = useState("table");
  const [version, setVersion] = useState(0); // bumps after every run to refetch views
  const [figure, setFigure] = useState<any>(null);
  const [page, setPage] = useState<TablePage | null>(null);
  const [pct, setPct] = useState<number | null>(null);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [failure, setFailure] = useState<Failure | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  const [picked, setPicked] = useState<number | null>(null);

  const stage = stages.find((s) => s.id === current);
  const hasResult = stage?.status === "done" || stage?.status === "stale";
  const tabs = stage ? [...stage.views, ...(stage.has_table ? ["table"] : [])] : [];
  const pickMode = stage?.ppm_views[tab];
  const onPick = pickMode
    ? (pt: PickPoint) => {
        const v = Number(pickMode === "customdata" ? (pt.customdata as unknown[] | undefined)?.[0] : pt.x);
        if (Number.isFinite(v)) setPicked(v);
      }
    : undefined;

  useEffect(() => {
    Api.connect().then(setApi);
    window.metbit.onBackendExit(() => setBackendDown(true));
  }, []);

  useEffect(() => {
    api?.stages().then(setStages);
  }, [api]);

  useEffect(() => {
    setFigure(null);
    setPage(null);
    if (!api || !hasResult) return;
    let live = true; // drop responses for a view the user already left
    const load = tab === "table" ? api.table(current).then((p) => live && setPage(p)) : api.figure(current, tab).then((f) => live && setFigure(f));
    load.catch((e: Error) => live && setFailure({ message: e.message }));
    return () => {
      live = false;
    };
  }, [api, current, tab, hasResult, version]);

  function select(id: string) {
    setCurrent(id);
    setErrors({});
    setFailure(null);
    setTab(stages.find((s) => s.id === id)?.views[0] ?? "table");
  }

  async function run(values: Values) {
    if (!api) return;
    setErrors({});
    setFailure(null);
    setPct(0.05);
    setStages((ss) => ss.map((s) => (s.id === current ? { ...s, status: "running" } : s)));
    try {
      await api.run(current, values, (e) => {
        if (e.event === "progress") setPct(e.data.pct);
        if (e.event === "error") setFailure(e.data);
      });
    } catch (e) {
      if (e instanceof ApiError && e.status === 422) setErrors(fieldErrors(e.detail));
      else setFailure({ message: (e as Error).message });
    }
    setPct(null);
    setStages(await api.stages());
    setVersion((v) => v + 1);
  }

  async function restart() {
    await window.metbit.restartBackend();
    setBackendDown(false);
    setApi(await Api.connect());
    select("import");
  }

  if (!stage) return <div class="placeholder">Starting metbit…</div>;

  let body;
  if (failure) {
    body = (
      <div class="placeholder">
        <div>
          <p class="error">{failure.message}</p>
          {failure.traceback && (
            <details>
              <summary>Details</summary>
              <pre>{failure.traceback}</pre>
            </details>
          )}
        </div>
      </div>
    );
  } else if (!hasResult) {
    body = <div class="placeholder">{stage.status === "running" ? "Running…" : "Set parameters and press Run"}</div>;
  } else if (tab === "table") {
    body = page && <DataTable page={page} />;
  } else {
    body = figure && <PlotView figure={figure} onPick={onPick} />;
  }

  return (
    <div class="layout">
      <JobBar pct={pct} />
      <Rail stages={stages} current={current} onSelect={select} />
      <main class="center">
        <div class="tabs" role="tablist">
          {hasResult &&
            tabs.map((t) => (
              <button key={t} role="tab" aria-selected={t === tab} onClick={() => setTab(t)}>
                {label(t)}
              </button>
            ))}
          {picked !== null && (
            <span class="picked" aria-live="polite">
              δ {picked.toFixed(3)} ppm
            </span>
          )}
        </div>
        <div class="view">{body}</div>
      </main>
      <ParamPanel
        key={current}
        title={stage.title}
        schema={stage.schema}
        initial={stage.params}
        errors={errors}
        busy={pct !== null}
        onRun={run}
        picked={picked}
      />
      {backendDown && (
        <div class="banner" role="alert">
          Backend stopped
          <button class="btn" onClick={restart}>
            Restart
          </button>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Styles**

Append to `GUI/renderer/theme.css`:

```css
.rail-section {
  color: var(--muted);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 12px var(--s1) 4px;
}
.hint { color: var(--muted); font-size: 11px; }
.pick { all: unset; cursor: pointer; color: var(--accent); font-size: 12px; align-self: flex-start; }
.pick:focus-visible { outline: 2px solid var(--accent); }
.tabs .picked { margin-left: auto; align-self: center; color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
```

- [ ] **Step 4: Typecheck, test, build, manual walkthrough**

Run: `cd GUI && npm run typecheck && npm test && npm start`
Expected: typecheck exits 0; `13 passed`. In the app: the rail shows DATA / ANALYSIS / STATISTICS headers; importing a CSV enables Normalize, Align and every analysis; OPLS-DA opens with Group A and Group B dropdowns preset to the first two groups; clicking a peak on the Import Spectra view shows `δ x.xxx ppm` in the tab bar, and on STOCSY the Anchor field offers `Use x.xxx ppm`; clicking on the OPLS-DA Scores view does not change the picked value.

- [ ] **Step 5: Commit**

```bash
git add GUI/renderer
git commit -m "feat(gui): click plots to pick ppm and wire Phase A stages into the app"
```

---

### Task 7: End-to-end Phase A path

**Files:**
- Modify (replace): `GUI/e2e/smoke.spec.ts`

**Interfaces:**
- Consumes: rail buttons by title (`Normalize`, `OPLS-DA`, `STOCSY`, `Volcano`), control ids `#f-path`, `#f-group_a`, `#f-group_b`, `#f-anchor_ppm`, tab names (`VIP`, `Volcano`), `.plot .main-svg`.
- Produces: `npm run e2e` runs 2 tests.

- [ ] **Step 1: Replace the e2e spec**

Replace `GUI/e2e/smoke.spec.ts` with:

```ts
import { _electron as electron, expect, test, type Page } from "@playwright/test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function makeCsv(): string {
  const ppm = Array.from({ length: 40 }, (_, i) => (10 - i * 0.25).toFixed(4));
  const rows = Array.from({ length: 12 }, (_, i) => {
    const group = i < 6 ? "A" : "B";
    const values = ppm.map((_, j) => (Math.abs(Math.sin(i * 7 + j)) + 1 + (group === "B" && j === 5 ? 3 : 0)).toFixed(5));
    return [`s${i}`, group, ...values].join(",");
  });
  const file = join(mkdtempSync(join(tmpdir(), "metbit-e2e-")), "data.csv");
  writeFileSync(file, [["Sample", "Group", ...ppm].join(","), ...rows].join("\n"));
  return file;
}

const run = (win: Page) => win.getByRole("button", { name: "Run", exact: true }).click();
const stageButton = (win: Page, name: string) => win.locator(".rail button", { hasText: name });
const plot = (win: Page) => expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

test("import a CSV and run PCA", async () => {
  const app = await electron.launch({ args: ["."] });
  const win = await app.firstWindow();

  await win.fill("#f-path", makeCsv());
  await run(win);
  await expect(win.locator("table.data")).toContainText("s11");

  await stageButton(win, "PCA").click();
  await run(win);
  await plot(win);

  await app.close();
});

test("normalize, OPLS-DA, STOCSY and volcano", async () => {
  const app = await electron.launch({ args: ["."] });
  const win = await app.firstWindow();

  await win.fill("#f-path", makeCsv());
  await run(win);
  await expect(stageButton(win, "Normalize")).toBeEnabled();

  await stageButton(win, "Normalize").click();
  await run(win);
  await plot(win);

  await stageButton(win, "OPLS-DA").click();
  await expect(win.locator("#f-group_a")).toHaveValue("A");
  await expect(win.locator("#f-group_b")).toHaveValue("B");
  await run(win);
  await plot(win);
  await win.getByRole("tab", { name: "VIP" }).click();
  await plot(win);

  await stageButton(win, "STOCSY").click();
  await win.fill("#f-anchor_ppm", "5");
  await run(win);
  await plot(win);

  await stageButton(win, "Volcano").click();
  await run(win);
  await expect(win.getByRole("tab", { name: "Volcano" })).toBeVisible();
  await plot(win);

  await app.close();
});
```

- [ ] **Step 2: Run it**

Run: `cd GUI && npm run e2e`
Expected: `2 passed`.

- [ ] **Step 3: Full verification**

Run from the repo root:
```bash
python3 -m pytest -q -p no:cacheprovider && (cd GUI && npm run typecheck && npm test && npm run e2e)
```
Expected: repo suite passes with coverage at or above 99%; typecheck exits 0; `13 passed`; `2 passed`.

- [ ] **Step 4: Commit**

```bash
git add GUI/e2e
git commit -m "test(gui): cover the Phase A pipeline end to end"
```
