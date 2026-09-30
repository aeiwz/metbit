# metbit GUI Stage 1 (Shell and Bridge) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A runnable Electron desktop app that starts a local metbit Python backend, imports a CSV spectral matrix, runs PCA, and shows interactive Plotly results in a minimal three-column UI.

**Architecture:** Electron main spawns a FastAPI sidecar (`python -m metbit_server` in dev) on a random localhost port with a per-session bearer token, waits for `/health`, then opens a sandboxed renderer. The Preact renderer talks to the sidecar over `fetch`; stage outputs stay in Python memory and only Plotly figure JSON and small tables cross the wire. Spec: `GUI/docs/specs/2026-09-29-metbit-gui-design.md`.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, Pydantic v2, metbit, pytest + FastAPI TestClient. Node 20+, Electron, Vite, Preact, TypeScript, plotly.js-dist-min, @fontsource-variable/inter, Vitest + jsdom, Playwright (Electron).

## Global Constraints

- All GUI code lives under `GUI/` at the repo root (`GUI/electron`, `GUI/renderer`, `GUI/server`, `GUI/e2e`). Python GUI tests live in `tests/gui/`.
- Never use the em dash character or the section sign in any file, comment, or commit message.
- Commit messages carry no `Co-Authored-By` trailer.
- Keep every source file under 500 lines.
- Sidecar binds `127.0.0.1` only. Every route requires `Authorization: Bearer <token>`; token is passed to the sidecar via the `METBIT_TOKEN` env var, never argv.
- Renderer: `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`.
- Health check timeout: 20 s. Server log: `userData/logs/server.log`, rotated at 5 MB.
- Visual language: monochrome base, single accent `#2563EB`, Inter, 13 px base, 8 px spacing grid, 1 px hairline borders, no shadows, light and dark via `prefers-color-scheme`.
- Stage statuses (exact strings): `empty`, `running`, `done`, `failed`, `stale`.
- Python tests for the GUI run with `python3 -m pytest tests/gui --no-cov -q` (the repo-wide `--cov-fail-under=99` targets the `metbit` package only).

## File Map

```
GUI/
  package.json                 npm scripts and deps
  tsconfig.json                renderer typecheck (noEmit)
  tsconfig.electron.json       compiles electron/ to dist-electron/ (CommonJS)
  vite.config.mts              renderer build (root: renderer, out: dist)
  vitest.config.mts            unit tests for renderer and electron helpers
  playwright.config.ts         e2e config
  electron/
    sidecar.ts                 freePort, waitForHealth, openLog, startSidecar
    sidecar.test.ts            waitForHealth test
    main.ts                    app lifecycle, window, IPC
    preload.ts                 contextBridge API (window.metbit)
  renderer/
    index.html                 CSP + mount point
    main.tsx                   entry
    types.d.ts                 window.metbit typing, plotly module shim
    theme.css                  design tokens and layout
    api.ts                     HTTP client, SSE parser, shared types
    api.test.ts                parseSse test
    App.tsx                    state and run flow
    components/
      Rail.tsx                 pipeline rail with status dots
      ParamPanel.tsx           schema-driven parameter form
      ParamPanel.test.tsx
      PlotView.tsx             plotly.js wrapper
      DataTable.tsx            summary table
      JobBar.tsx               top progress bar
  e2e/
    smoke.spec.ts              launch, import CSV, run PCA, see plot
  server/
    requirements.txt
    metbit_server/
      __init__.py
      __main__.py              CLI entry (uvicorn)
      session.py               in-memory pipeline state
      stages.py                stage registry: Import (CSV), PCA
      jobs.py                  worker thread + event queue
      app.py                   FastAPI routes
tests/gui/
  conftest.py                  sys.path, fixtures
  test_gui_stages.py
  test_gui_app.py
  test_gui_views.py
```

---

### Task 1: Session state and stage registry (Import CSV, PCA)

**Files:**
- Create: `GUI/server/requirements.txt`
- Create: `GUI/server/metbit_server/__init__.py`
- Create: `GUI/server/metbit_server/session.py`
- Create: `GUI/server/metbit_server/stages.py`
- Create: `tests/gui/conftest.py`
- Test: `tests/gui/test_gui_stages.py`
- Modify: `requirements-dev.txt` (append FastAPI deps)

**Interfaces:**
- Consumes: `metbit.pca` (`pca(X, label, features_name, n_components, scaling_method)`, `.fit()`, `.plot_pca_scores(pc=[..])`, `.plot_loading_(pc=[..])`, `.plot_observe_variance()`, `.get_explained_variance()`).
- Produces:
  - `session.Session(stage_ids: list[str])` with attributes `order`, `status: dict[str, str]`, `params: dict[str, dict]`, `outputs: dict[str, Any]`, `lock: threading.Lock`; methods `start(sid)`, `complete(sid, params: dict, output)`, `fail(sid)`, `output(sid) -> Any` (raises `KeyError`).
  - `stages.Stage` dataclass: `id: str`, `title: str`, `params: type[BaseModel]`, `run: Callable[[BaseModel, Session, Progress], Any]`, `table: Callable[[Any], pd.DataFrame]`, `figures: dict[str, Callable[[Any], go.Figure]]`, `requires: str | None`.
  - `stages.Progress = Callable[[float], None]` (fraction 0..1).
  - `stages.STAGES: list[Stage]` = `[import, pca]`.
  - `stages.ImportParams(path: str, group_column: str = "Group")`, `stages.Dataset(X, meta)`.
  - `stages.PCAParams(n_components=2, scaling="pareto", pc_x=1, pc_y=2)`, `stages.PCAResult(model, pcs: list[str])`.
  - PCA figure view names: `"scores"`, `"loadings"`, `"variance"`.
  - JSON schema of `ImportParams.path` carries `"format": "file-path"`.

- [ ] **Step 1: Add dependencies**

Create `GUI/server/requirements.txt`:

```
metbit
fastapi>=0.110
uvicorn>=0.29
```

Append to `requirements-dev.txt`:

```

# GUI sidecar tests (tests/gui)
fastapi>=0.110
uvicorn>=0.29
httpx
```

Run: `python3 -c "import fastapi, uvicorn, httpx; print('ok')"`
Expected: `ok` (if not, `python3 -m pip install fastapi uvicorn httpx`).

- [ ] **Step 2: Write test fixtures**

Create `GUI/server/metbit_server/__init__.py` as an empty file.

Create `tests/gui/conftest.py`:

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
    X = rng.normal(size=(12, 40))
    X[6:, 5] += 3  # group B carries an extra peak
    df = pd.DataFrame(X, columns=[f"{v:.4f}" for v in ppm])
    df.insert(0, "Group", ["A"] * 6 + ["B"] * 6)
    df.insert(0, "Sample", [f"s{i}" for i in range(12)])
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)
```

- [ ] **Step 3: Write the failing tests**

Create `tests/gui/test_gui_stages.py`:

```python
import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from metbit_server.session import Session
from metbit_server.stages import STAGES, ImportParams, PCAParams

BY_ID = {s.id: s for s in STAGES}


def _noop(_pct):
    pass


def _session_with_import(csv_path):
    s = Session([st.id for st in STAGES])
    p = ImportParams(path=csv_path)
    s.complete("import", p.model_dump(), BY_ID["import"].run(p, s, _noop))
    return s


def test_registry_order_and_requirements():
    assert [s.id for s in STAGES] == ["import", "pca"]
    assert BY_ID["pca"].requires == "import"
    assert ImportParams.model_json_schema()["properties"]["path"]["format"] == "file-path"


def test_import_splits_features_and_metadata(csv_path):
    s = Session(["import", "pca"])
    data = BY_ID["import"].run(ImportParams(path=csv_path), s, _noop)
    assert data.X.shape == (12, 40)
    assert list(data.meta.columns) == ["Sample", "Group"]
    assert list(BY_ID["import"].table(data).columns) == ["Sample", "Group"]


def test_import_rejects_missing_group_column(csv_path):
    with pytest.raises(ValueError, match="Group column 'Class' not found"):
        BY_ID["import"].run(ImportParams(path=csv_path, group_column="Class"), Session(["import"]), _noop)


def test_import_rejects_csv_without_ppm_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Sample,Group\ns1,A\n")
    with pytest.raises(ValueError, match="No spectral columns"):
        BY_ID["import"].run(ImportParams(path=str(bad)), Session(["import"]), _noop)


def test_pca_run_figures_and_table(csv_path):
    s = _session_with_import(csv_path)
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


def test_session_complete_marks_later_done_stages_stale():
    s = Session(["import", "pca"])
    s.complete("import", {}, 1)
    s.complete("pca", {}, 2)
    s.complete("import", {}, 3)
    assert s.status == {"import": "done", "pca": "stale"}
    with pytest.raises(KeyError):
        Session(["import"]).output("import")
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui --no-cov -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'metbit_server.session'`.

- [ ] **Step 5: Implement `session.py`**

Create `GUI/server/metbit_server/session.py`:

```python
"""In-memory pipeline state for one GUI session."""
import threading
from typing import Any


class Session:
    def __init__(self, stage_ids: list[str]) -> None:
        self.order = list(stage_ids)
        self.status: dict[str, str] = {sid: "empty" for sid in self.order}
        self.params: dict[str, dict] = {}
        self.outputs: dict[str, Any] = {}
        self.lock = threading.Lock()  # one job at a time

    def start(self, sid: str) -> None:
        self.status[sid] = "running"

    def complete(self, sid: str, params: dict, output: Any) -> None:
        self.params[sid] = params
        self.outputs[sid] = output
        self.status[sid] = "done"
        # ponytail: stages are linear, so "downstream" means "later in order"
        for later in self.order[self.order.index(sid) + 1:]:
            if self.status[later] == "done":
                self.status[later] = "stale"

    def fail(self, sid: str) -> None:
        self.status[sid] = "failed"

    def output(self, sid: str) -> Any:
        if sid not in self.outputs:
            raise KeyError(sid)
        return self.outputs[sid]
```

- [ ] **Step 6: Implement `stages.py`**

Create `GUI/server/metbit_server/stages.py`:

```python
"""Pipeline stage registry: params schema, run function, figures and table per stage."""
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field, model_validator

from metbit import pca

from .session import Session

Progress = Callable[[float], None]


@dataclass
class Stage:
    id: str
    title: str
    params: type[BaseModel]
    run: Callable[[Any, Session, Progress], Any]
    table: Callable[[Any], pd.DataFrame]
    figures: dict[str, Callable[[Any], go.Figure]] = field(default_factory=dict)
    requires: Optional[str] = None


# Import (CSV) ---------------------------------------------------------------

class ImportParams(BaseModel):
    path: str = Field(title="CSV file", json_schema_extra={"format": "file-path"})
    group_column: str = Field("Group", title="Group column")


@dataclass
class Dataset:
    X: pd.DataFrame     # samples x features, ppm column headers
    meta: pd.DataFrame  # every non-ppm column


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
    progress(1.0)
    return Dataset(X=X, meta=meta)


# PCA ------------------------------------------------------------------------

class PCAParams(BaseModel):
    n_components: int = Field(2, ge=2, le=10, title="Components")
    scaling: Literal["pareto", "uv", "mean"] = Field("pareto", title="Scaling")
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
    data: Dataset = session.output("import")
    groups = data.meta[session.params["import"]["group_column"]].astype(str)
    model = pca(
        X=data.X,
        label=groups,
        features_name=list(data.X.columns.astype(float)),
        n_components=p.n_components,
        scaling_method=p.scaling,
    )
    model.fit()
    progress(1.0)
    return PCAResult(model=model, pcs=[f"PC{p.pc_x}", f"PC{p.pc_y}"])


STAGES: list[Stage] = [
    Stage(
        id="import",
        title="Import",
        params=ImportParams,
        run=run_import,
        table=lambda d: d.meta,
    ),
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
    ),
]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q`
Expected: `7 passed`.

- [ ] **Step 8: Commit**

```bash
git add GUI/server tests/gui requirements-dev.txt
git commit -m "feat(gui): add sidecar session state and Import/PCA stage registry"
```

---

### Task 2: FastAPI app, auth, job runner, run + SSE events

**Files:**
- Create: `GUI/server/metbit_server/jobs.py`
- Create: `GUI/server/metbit_server/app.py`
- Create: `GUI/server/metbit_server/__main__.py`
- Test: `tests/gui/test_gui_app.py`

**Interfaces:**
- Consumes: `Session`, `Stage`, `STAGES` from Task 1.
- Produces:
  - `jobs.Busy(Exception)`, `jobs.Job(id: str, events: queue.Queue[tuple[str, dict]])`, `jobs.start_job(session, stage, params: BaseModel) -> Job`.
  - `app.create_app(token: str, stages: list[Stage] = STAGES) -> FastAPI`.
  - HTTP: `GET /health -> {"ok": true}`; `GET /stages -> [{id, title, requires, status, views, schema, params}]`; `POST /stages/{id}/run` (JSON body = params) `-> {"job_id"}`, 404 unknown stage, 422 invalid params (`detail` = list of Pydantic errors with `loc`, `msg`), 409 when `requires` stage is not `done` or another job runs; `GET /jobs/{job_id}/events` -> `text/event-stream` with events `progress {"pct"}`, `done {"stage"}`, `error {"message","traceback"}`; stream ends after `done` or `error`.
  - CLI: `METBIT_TOKEN=<token> python -m metbit_server --port <port>` (run with cwd `GUI/server`).

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_app.py`:

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
    stages = client.get("/stages", headers=AUTH).json()
    assert [s["id"] for s in stages] == ["import", "pca"]
    assert stages[0]["status"] == "empty"
    assert stages[0]["schema"]["properties"]["path"]["format"] == "file-path"
    assert stages[1]["views"] == ["scores", "loadings", "variance"]


def test_import_then_pca_then_reimport_marks_stale(client, csv_path):
    assert run_stage(client, "import", {"path": csv_path})[-1] == "done"
    assert run_stage(client, "pca", {"n_components": 2})[-1] == "done"
    assert statuses(client) == {"import": "done", "pca": "done"}
    run_stage(client, "import", {"path": csv_path})
    assert statuses(client) == {"import": "done", "pca": "stale"}
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

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_app.py --no-cov -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'metbit_server.app'`.

- [ ] **Step 3: Implement `jobs.py`**

Create `GUI/server/metbit_server/jobs.py`:

```python
"""Run one stage at a time on a worker thread and queue its events for SSE."""
import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel


class Busy(Exception):
    """Another job holds the session lock."""


@dataclass
class Job:
    id: str
    events: "queue.Queue[tuple[str, dict]]" = field(default_factory=queue.Queue)


def start_job(session, stage, params: BaseModel) -> Job:
    if not session.lock.acquire(blocking=False):
        raise Busy()
    job = Job(uuid.uuid4().hex)
    session.start(stage.id)

    def progress(pct: float) -> None:
        job.events.put(("progress", {"pct": round(float(pct), 3)}))

    def work() -> None:
        try:
            out = stage.run(params, session, progress)
            session.complete(stage.id, params.model_dump(), out)
            event = ("done", {"stage": stage.id})
        except Exception as e:
            session.fail(stage.id)
            event = ("error", {"message": str(e), "traceback": traceback.format_exc()})
        # release before announcing, so a client reacting to "done" can run again at once
        session.lock.release()
        job.events.put(event)

    threading.Thread(target=work, daemon=True).start()
    return job
```

- [ ] **Step 4: Implement `app.py`**

Create `GUI/server/metbit_server/app.py`:

```python
"""FastAPI sidecar for the metbit desktop GUI."""
import asyncio
import json
import secrets

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from .jobs import Busy, Job, start_job
from .session import Session
from .stages import STAGES, Stage


def create_app(token: str, stages: list[Stage] = STAGES) -> FastAPI:
    by_id = {s.id: s for s in stages}
    session = Session([s.id for s in stages])
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

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/stages")
    def list_stages():
        return [
            {
                "id": s.id,
                "title": s.title,
                "requires": s.requires,
                "status": session.status[s.id],
                "views": list(s.figures),
                "schema": s.params.model_json_schema(),
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

    return app
```

- [ ] **Step 5: Implement `__main__.py`**

Create `GUI/server/metbit_server/__main__.py`:

```python
"""Entry point: METBIT_TOKEN=<token> python -m metbit_server --port <port>"""
import argparse
import os
import sys

import uvicorn

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="metbit_server")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    token = os.environ.get("METBIT_TOKEN")
    if not token:
        sys.exit("METBIT_TOKEN is not set")
    uvicorn.run(create_app(token), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q`
Expected: `16 passed`.

- [ ] **Step 7: Smoke the CLI**

Run (from repo root):
```bash
cd GUI/server && (METBIT_TOKEN=abc PYTHONPATH=../.. python3 -m metbit_server --port 8765 & echo $! > /tmp/mbs.pid) && sleep 3 && curl -s -H "Authorization: Bearer abc" http://127.0.0.1:8765/health; kill $(cat /tmp/mbs.pid)
```
Expected: `{"ok":true}`

- [ ] **Step 8: Commit**

```bash
git add GUI/server tests/gui
git commit -m "feat(gui): add FastAPI sidecar with token auth, job runner and SSE events"
```

---

### Task 3: Figure and table endpoints

**Files:**
- Modify: `GUI/server/metbit_server/app.py` (add two routes before `return app`)
- Test: `tests/gui/test_gui_views.py`

**Interfaces:**
- Consumes: `create_app` from Task 2 (new routes close over its `session`, `by_id` and `stage_or_404`).
- Produces:
  - `GET /stages/{id}/figure?view=<name>` -> Plotly figure JSON (`{"data": [...], "layout": {...}}`) with `layout.width`/`layout.height` removed and `autosize: true`; 409 if stage has no result (status not `done`/`stale`); 404 unknown stage or view.
  - `GET /stages/{id}/table?offset=0&limit=100` (limit max 1000) -> `{"columns": [str], "rows": [[...]], "total": int}`; 409 if no result.

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_views.py`:

```python
import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app


@pytest.fixture
def client(csv_path):
    c = TestClient(create_app(TOKEN))
    for sid, params in (("import", {"path": csv_path}), ("pca", {"n_components": 3})):
        job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
        assert "event: done" in c.get(f"/jobs/{job}/events", headers=AUTH).text
    return c


def test_figure_is_plotly_json_without_fixed_size(client):
    fig = client.get("/stages/pca/figure?view=scores", headers=AUTH).json()
    assert fig["data"]
    assert "width" not in fig["layout"] and "height" not in fig["layout"]
    assert fig["layout"]["autosize"] is True


def test_unknown_view_is_404(client):
    assert client.get("/stages/pca/figure?view=nope", headers=AUTH).status_code == 404


def test_no_result_is_409():
    c = TestClient(create_app(TOKEN))
    assert c.get("/stages/pca/figure?view=scores", headers=AUTH).status_code == 409
    assert c.get("/stages/import/table", headers=AUTH).status_code == 409


def test_table_pagination(client):
    page = client.get("/stages/import/table?offset=10&limit=5", headers=AUTH).json()
    assert page["columns"] == ["Sample", "Group"]
    assert page["rows"] == [["s10", "B"], ["s11", "B"]]
    assert page["total"] == 12
    assert client.get("/stages/import/table?limit=5000", headers=AUTH).status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_views.py --no-cov -q`
Expected: FAIL, figure/table requests return 404 (routes do not exist).

- [ ] **Step 3: Implement the routes**

In `GUI/server/metbit_server/app.py`, change the fastapi import line to:

```python
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, Response
```

Then insert immediately before the final `return app`:

```python
    def result_or_409(sid: str):
        stage_or_404(sid)
        if session.status[sid] not in ("done", "stale"):
            raise HTTPException(409, f"Stage '{sid}' has no result")
        return session.outputs[sid]

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
        df = by_id[sid].table(result_or_409(sid))
        page = json.loads(df.iloc[offset:offset + limit].to_json(orient="split", index=False))
        return {"columns": [str(c) for c in page["columns"]], "rows": page["data"], "total": len(df)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q`
Expected: `20 passed`.

- [ ] **Step 5: Commit**

```bash
git add GUI/server tests/gui
git commit -m "feat(gui): add figure and table endpoints"
```

---

### Task 4: Electron shell, sidecar launcher, preload bridge

**Files:**
- Create: `GUI/package.json` (via npm), `GUI/tsconfig.json`, `GUI/tsconfig.electron.json`, `GUI/vite.config.mts`, `GUI/vitest.config.mts`
- Create: `GUI/electron/sidecar.ts`, `GUI/electron/main.ts`, `GUI/electron/preload.ts`
- Create: `GUI/renderer/index.html`, `GUI/renderer/main.tsx`, `GUI/renderer/types.d.ts`
- Test: `GUI/electron/sidecar.test.ts`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: sidecar CLI from Task 2 (`METBIT_TOKEN` env, `--port`, `GET /health`).
- Produces:
  - `sidecar.ts`: `freePort(): Promise<number>`, `waitForHealth(url, token, timeoutMs): Promise<void>` (throws `Backend not ready after <n> ms`), `openLog(dir): { file, stream }`, `startSidecar(cmd: SidecarCommand, logDir): Promise<Backend>` where `SidecarCommand = { cmd: string; args: string[]; cwd?: string; env?: Record<string, string> }` and `Backend = { port, token, proc, logFile }`.
  - Renderer global `window.metbit`: `backendInfo(): Promise<{port, token}>`, `restartBackend(): Promise<{port, token}>`, `openFile(filters?): Promise<string | null>`, `onBackendExit(cb): void`.
  - npm scripts: `build`, `start`, `test`, `typecheck`, `e2e`.
  - Dev sidecar command: `$METBIT_PYTHON` (default `python3`, `python` on Windows) `-m metbit_server`, cwd `GUI/server`, `PYTHONPATH` = repo root.

- [ ] **Step 1: Scaffold the package**

Run:
```bash
mkdir -p GUI/electron GUI/renderer/components GUI/e2e
cd GUI
npm init -y
npm pkg set name=metbit-gui version=0.1.0 private=true main=dist-electron/main.js
npm pkg set scripts.build:renderer="vite build" scripts.build:electron="tsc -p tsconfig.electron.json" \
  scripts.build="npm run build:renderer && npm run build:electron" \
  scripts.start="npm run build && electron ." scripts.test="vitest run" \
  scripts.typecheck="tsc -p tsconfig.json && tsc -p tsconfig.electron.json --noEmit" \
  scripts.e2e="npm run build && playwright test"
npm install preact plotly.js-dist-min @fontsource-variable/inter
npm install -D electron vite @preact/preset-vite typescript vitest jsdom @playwright/test @types/node
```

Append to repo-root `.gitignore`:

```
# GUI
/GUI/node_modules/
/GUI/dist-electron/
/GUI/test-results/
```

(`dist/` is already ignored repo-wide, which covers `GUI/dist/`.)

- [ ] **Step 2: Add TypeScript, Vite and Vitest config**

Create `GUI/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "jsx": "react-jsx",
    "jsxImportSource": "preact",
    "strict": true,
    "noEmit": true,
    "skipLibCheck": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "types": ["vite/client"]
  },
  "include": ["renderer"]
}
```

Create `GUI/tsconfig.electron.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "CommonJS",
    "moduleResolution": "Node",
    "outDir": "dist-electron",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "types": ["node"]
  },
  "include": ["electron"],
  "exclude": ["electron/**/*.test.ts"]
}
```

Create `GUI/vite.config.mts`:

```ts
import { defineConfig } from "vite";
import preact from "@preact/preset-vite";

export default defineConfig({
  root: "renderer",
  base: "./", // loaded via file:// in Electron
  plugins: [preact()],
  build: { outDir: "../dist", emptyOutDir: true },
});
```

Create `GUI/vitest.config.mts`:

```ts
import { defineConfig } from "vitest/config";
import preact from "@preact/preset-vite";

export default defineConfig({
  plugins: [preact()],
  test: {
    environment: "jsdom",
    include: ["renderer/**/*.test.{ts,tsx}", "electron/**/*.test.ts"],
  },
});
```

- [ ] **Step 3: Write the failing test**

Create `GUI/electron/sidecar.test.ts`:

```ts
// @vitest-environment node
import { createServer } from "node:http";
import { afterEach, expect, it } from "vitest";
import { freePort, waitForHealth } from "./sidecar";

let close: (() => void) | undefined;
afterEach(() => close?.());

it("resolves once /health answers 200 with the right token", async () => {
  const port = await freePort();
  const srv = createServer((req, res) => {
    const ok = req.url === "/health" && req.headers.authorization === "Bearer t";
    res.writeHead(ok ? 200 : 401).end();
  });
  await new Promise<void>((r) => srv.listen(port, "127.0.0.1", r));
  close = () => srv.close();
  await expect(waitForHealth(`http://127.0.0.1:${port}`, "t", 2000)).resolves.toBeUndefined();
});

it("rejects after the timeout when nothing listens", async () => {
  const port = await freePort();
  await expect(waitForHealth(`http://127.0.0.1:${port}`, "t", 500)).rejects.toThrow("Backend not ready after 500 ms");
});
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd GUI && npx vitest run electron`
Expected: FAIL, cannot resolve `./sidecar`.

- [ ] **Step 5: Implement `sidecar.ts`**

Create `GUI/electron/sidecar.ts`:

```ts
import { spawn, type ChildProcess } from "node:child_process";
import { randomBytes } from "node:crypto";
import { createWriteStream, existsSync, mkdirSync, renameSync, statSync, type WriteStream } from "node:fs";
import { createServer } from "node:net";
import { join } from "node:path";

export interface SidecarCommand {
  cmd: string;
  args: string[];
  cwd?: string;
  env?: Record<string, string>;
}

export interface Backend {
  port: number;
  token: string;
  proc: ChildProcess;
  logFile: string;
}

export function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.once("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address() as { port: number };
      srv.close(() => resolve(port));
    });
  });
}

export async function waitForHealth(url: string, token: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`${url}/health`, { headers: { Authorization: `Bearer ${token}` } });
      if (r.ok) return;
    } catch {
      // not listening yet
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`Backend not ready after ${timeoutMs} ms`);
}

const MAX_LOG_BYTES = 5 * 1024 * 1024;

export function openLog(dir: string): { file: string; stream: WriteStream } {
  mkdirSync(dir, { recursive: true });
  const file = join(dir, "server.log");
  // ponytail: one backup file; add numbered rotation if anyone needs older logs
  if (existsSync(file) && statSync(file).size > MAX_LOG_BYTES) renameSync(file, `${file}.1`);
  return { file, stream: createWriteStream(file, { flags: "a" }) };
}

export async function startSidecar(command: SidecarCommand, logDir: string): Promise<Backend> {
  const port = await freePort();
  const token = randomBytes(24).toString("hex");
  const log = openLog(logDir);
  const proc = spawn(command.cmd, [...command.args, "--port", String(port)], {
    cwd: command.cwd,
    env: { ...process.env, ...command.env, METBIT_TOKEN: token },
  });
  proc.stdout?.pipe(log.stream);
  proc.stderr?.pipe(log.stream);
  try {
    await waitForHealth(`http://127.0.0.1:${port}`, token, 20_000);
  } catch (e) {
    proc.kill();
    throw e;
  }
  return { port, token, proc, logFile: log.file };
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd GUI && npx vitest run electron`
Expected: `2 passed`.

- [ ] **Step 7: Implement main process and preload**

Create `GUI/electron/main.ts`:

```ts
import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { join } from "node:path";
import { startSidecar, type Backend, type SidecarCommand } from "./sidecar";

let backend: Backend | null = null;
let win: BrowserWindow | null = null;
const logDir = () => join(app.getPath("userData"), "logs");

function sidecarCommand(): SidecarCommand {
  if (app.isPackaged) {
    const exe = process.platform === "win32" ? "metbit-server.exe" : "metbit-server";
    return { cmd: join(process.resourcesPath, "metbit-server", exe), args: [] };
  }
  // dev: run the sidecar from source with a Python that has metbit's deps
  const serverDir = join(__dirname, "..", "server");
  return {
    cmd: process.env.METBIT_PYTHON ?? (process.platform === "win32" ? "python" : "python3"),
    args: ["-m", "metbit_server"],
    cwd: serverDir,
    env: { PYTHONPATH: join(serverDir, "..", "..") },
  };
}

async function launchBackend(): Promise<Backend> {
  const b = await startSidecar(sidecarCommand(), logDir());
  b.proc.on("exit", () => {
    if (backend === b) win?.webContents.send("backend:exit"); // ignore deliberate restarts
  });
  backend = b;
  return b;
}

const info = (b: Backend) => ({ port: b.port, token: b.token });

ipcMain.handle("backend:info", () => (backend ? info(backend) : null));
ipcMain.handle("backend:restart", async () => {
  const old = backend;
  backend = null;
  old?.proc.kill();
  return info(await launchBackend());
});
ipcMain.handle("dialog:openFile", async (_e, filters?: Electron.FileFilter[]) => {
  const r = await dialog.showOpenDialog(win!, { properties: ["openFile"], filters });
  return r.canceled ? null : r.filePaths[0];
});

async function boot() {
  try {
    await launchBackend();
  } catch (e) {
    dialog.showErrorBox("metbit could not start", `${(e as Error).message}\n\nLog: ${join(logDir(), "server.log")}`);
    app.quit();
    return;
  }
  win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 960,
    minHeight: 600,
    title: "metbit",
    webPreferences: {
      preload: join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  win.loadFile(join(__dirname, "..", "dist", "index.html"));
}

app.whenReady().then(boot);
app.on("window-all-closed", () => app.quit());
app.on("will-quit", () => backend?.proc.kill());
```

Create `GUI/electron/preload.ts`:

```ts
import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("metbit", {
  backendInfo: () => ipcRenderer.invoke("backend:info"),
  restartBackend: () => ipcRenderer.invoke("backend:restart"),
  openFile: (filters?: { name: string; extensions: string[] }[]) => ipcRenderer.invoke("dialog:openFile", filters),
  onBackendExit: (cb: () => void) => {
    ipcRenderer.on("backend:exit", () => cb());
  },
});
```

- [ ] **Step 8: Minimal renderer that proves the bridge**

Create `GUI/renderer/types.d.ts`:

```ts
interface BackendInfo {
  port: number;
  token: string;
}

interface Window {
  metbit: {
    backendInfo(): Promise<BackendInfo>;
    restartBackend(): Promise<BackendInfo>;
    openFile(filters?: { name: string; extensions: string[] }[]): Promise<string | null>;
    onBackendExit(cb: () => void): void;
  };
}

declare module "plotly.js-dist-min";
```

Create `GUI/renderer/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <!-- unsafe-eval: plotly's WebGL traces (regl) compile shaders with new Function -->
    <meta
      http-equiv="Content-Security-Policy"
      content="default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src http://127.0.0.1:*"
    />
    <title>metbit</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="./main.tsx"></script>
  </body>
</html>
```

Create `GUI/renderer/main.tsx` (replaced in Task 7):

```tsx
import { render } from "preact";

async function probe(): Promise<string> {
  const { port, token } = await window.metbit.backendInfo();
  const r = await fetch(`http://127.0.0.1:${port}/health`, { headers: { Authorization: `Bearer ${token}` } });
  return r.ok ? `Backend ready on port ${port}` : `Backend error ${r.status}`;
}

probe().then((msg) => render(<p>{msg}</p>, document.getElementById("app")!));
```

- [ ] **Step 9: Build, typecheck and launch**

Run: `cd GUI && npm run typecheck && npm start`
Expected: typecheck exits 0; a window opens showing `Backend ready on port <n>`. Close the window; confirm no `metbit_server` process remains (`pgrep -f metbit_server` prints nothing).

- [ ] **Step 10: Commit**

```bash
git add .gitignore GUI/package.json GUI/package-lock.json GUI/tsconfig.json GUI/tsconfig.electron.json \
  GUI/vite.config.mts GUI/vitest.config.mts GUI/electron GUI/renderer
git commit -m "feat(gui): add Electron shell that launches the sidecar and bridges it to the renderer"
```

---

### Task 5: Theme, API client, pipeline rail

**Files:**
- Create: `GUI/renderer/theme.css`
- Create: `GUI/renderer/api.ts`
- Create: `GUI/renderer/components/Rail.tsx`
- Test: `GUI/renderer/api.test.ts`

**Interfaces:**
- Consumes: HTTP API from Tasks 2 and 3; `window.metbit.backendInfo()` from Task 4.
- Produces (`api.ts`):
  - `type Status = "empty" | "running" | "done" | "failed" | "stale"`
  - `interface FieldSchema { type?: string; title?: string; default?: unknown; enum?: string[]; minimum?: number; maximum?: number; format?: string }`
  - `interface JsonSchema { properties: Record<string, FieldSchema>; required?: string[] }`
  - `interface StageInfo { id: string; title: string; requires: string | null; status: Status; views: string[]; schema: JsonSchema; params: Record<string, unknown> | null }`
  - `interface TablePage { columns: string[]; rows: unknown[][]; total: number }`
  - `interface SseEvent { event: string; data: any }`
  - `class ApiError extends Error { status: number; detail: unknown }`
  - `parseSse(buffer: string): { events: SseEvent[]; rest: string }`
  - `class Api { static connect(): Promise<Api>; stages(): Promise<StageInfo[]>; figure(id, view): Promise<any>; table(id, offset?, limit?): Promise<TablePage>; run(id, params, onEvent: (e: SseEvent) => void): Promise<void> }`
- Produces (`Rail.tsx`): `Rail({ stages: StageInfo[]; current: string; onSelect: (id: string) => void })`.
- CSS classes used later: `.layout`, `.jobbar`, `.rail`, `.dot.<status>`, `.center`, `.tabs`, `.view`, `.empty`, `.panel`, `.field`, `.field.invalid`, `.err`, `.row`, `.btn`, `.btn.primary`, `.error`, `.banner`, `table.data`, `.muted`, `.plot`.

- [ ] **Step 1: Write the failing test**

Create `GUI/renderer/api.test.ts`:

```ts
import { expect, it } from "vitest";
import { parseSse } from "./api";

it("parses complete events and keeps the partial remainder", () => {
  const buf = 'event: progress\ndata: {"pct":0.5}\n\nevent: done\ndata: {"stage":"pca"}\n\nevent: err';
  const { events, rest } = parseSse(buf);
  expect(events).toEqual([
    { event: "progress", data: { pct: 0.5 } },
    { event: "done", data: { stage: "pca" } },
  ]);
  expect(rest).toBe("event: err");
});

it("returns nothing for an empty buffer", () => {
  expect(parseSse("")).toEqual({ events: [], rest: "" });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd GUI && npx vitest run renderer/api.test.ts`
Expected: FAIL, cannot resolve `./api`.

- [ ] **Step 3: Implement `api.ts`**

Create `GUI/renderer/api.ts`:

```ts
export type Status = "empty" | "running" | "done" | "failed" | "stale";

export interface FieldSchema {
  type?: string;
  title?: string;
  default?: unknown;
  enum?: string[];
  minimum?: number;
  maximum?: number;
  format?: string;
}

export interface JsonSchema {
  properties: Record<string, FieldSchema>;
  required?: string[];
}

export interface StageInfo {
  id: string;
  title: string;
  requires: string | null;
  status: Status;
  views: string[];
  schema: JsonSchema;
  params: Record<string, unknown> | null;
}

export interface TablePage {
  columns: string[];
  rows: unknown[][];
  total: number;
}

export interface SseEvent {
  event: string;
  data: any;
}

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : `HTTP ${status}`);
  }
}

export function parseSse(buffer: string): { events: SseEvent[]; rest: string } {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events = blocks.filter(Boolean).map((block) => {
    let event = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7);
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    return { event, data: data ? JSON.parse(data) : null };
  });
  return { events, rest };
}

export class Api {
  constructor(private base: string, private token: string) {}

  static async connect(): Promise<Api> {
    const { port, token } = await window.metbit.backendInfo();
    return new Api(`http://127.0.0.1:${port}`, token);
  }

  private async req(path: string, init: RequestInit = {}): Promise<Response> {
    const r = await fetch(this.base + path, {
      ...init,
      headers: { Authorization: `Bearer ${this.token}`, "Content-Type": "application/json" },
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new ApiError(r.status, body.detail ?? r.statusText);
    }
    return r;
  }

  stages(): Promise<StageInfo[]> {
    return this.req("/stages").then((r) => r.json());
  }

  figure(id: string, view: string): Promise<any> {
    return this.req(`/stages/${id}/figure?view=${encodeURIComponent(view)}`).then((r) => r.json());
  }

  table(id: string, offset = 0, limit = 100): Promise<TablePage> {
    return this.req(`/stages/${id}/table?offset=${offset}&limit=${limit}`).then((r) => r.json());
  }

  async run(id: string, params: Record<string, unknown>, onEvent: (e: SseEvent) => void): Promise<void> {
    const { job_id } = await this.req(`/stages/${id}/run`, {
      method: "POST",
      body: JSON.stringify(params),
    }).then((r) => r.json());
    // fetch, not EventSource: EventSource cannot send the Authorization header
    const r = await this.req(`/jobs/${job_id}/events`);
    const reader = r.body!.pipeThrough(new TextDecoderStream()).getReader();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      const parsed = parseSse(buf + value);
      buf = parsed.rest;
      parsed.events.forEach(onEvent);
    }
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd GUI && npx vitest run renderer/api.test.ts`
Expected: `2 passed`.

- [ ] **Step 5: Add the theme**

Create `GUI/renderer/theme.css`:

```css
:root {
  --bg: #ffffff;
  --surface: #f5f5f5;
  --border: #e5e5e5;
  --text: #171717;
  --muted: #737373;
  --accent: #2563eb;
  --danger: #dc2626;
  --font: "Inter Variable", system-ui, sans-serif;
  --s1: 8px;
  --s2: 16px;
  --s3: 24px;
  color-scheme: light dark;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0a0a0a;
    --surface: #171717;
    --border: #262626;
    --text: #ededed;
    --muted: #a3a3a3;
    --accent: #3b82f6;
    --danger: #f87171;
  }
}

* { box-sizing: border-box; }
html, body, #app { height: 100%; margin: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font: 13px/1.5 var(--font);
  -webkit-font-smoothing: antialiased;
}

.layout {
  display: grid;
  grid-template-columns: 200px minmax(0, 1fr) 280px;
  grid-template-rows: 2px minmax(0, 1fr);
  height: 100%;
}

.jobbar { grid-column: 1 / -1; }
.jobbar > div { height: 2px; background: var(--accent); transition: width 0.2s, opacity 0.3s; }

.rail {
  border-right: 1px solid var(--border);
  padding: var(--s2) var(--s1);
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.rail-brand { font-weight: 600; letter-spacing: -0.01em; padding: 0 var(--s1) var(--s2); }
.rail button {
  all: unset;
  display: flex;
  align-items: center;
  gap: var(--s1);
  padding: 6px var(--s1);
  border-radius: 6px;
  color: var(--muted);
  cursor: pointer;
}
.rail button[aria-current="true"] { color: var(--text); background: var(--surface); }
.rail button:disabled { opacity: 0.4; cursor: default; }
.rail button:focus-visible { outline: 2px solid var(--accent); }

.dot { width: 8px; height: 8px; border-radius: 50%; border: 1.5px solid var(--muted); flex: none; }
.dot.done { background: var(--accent); border-color: var(--accent); }
.dot.running { background: var(--accent); border-color: var(--accent); animation: pulse 1s ease-in-out infinite; }
.dot.failed { background: var(--danger); border-color: var(--danger); }
.dot.stale { border-color: var(--accent); }
@keyframes pulse { 50% { opacity: 0.3; } }

.center { display: flex; flex-direction: column; min-width: 0; }
.tabs { display: flex; gap: var(--s2); padding: var(--s1) var(--s2); border-bottom: 1px solid var(--border); min-height: 41px; }
.tabs button { all: unset; cursor: pointer; color: var(--muted); padding: 4px 0; border-bottom: 1.5px solid transparent; }
.tabs button[aria-selected="true"] { color: var(--text); border-color: var(--text); }
.tabs button:focus-visible { outline: 2px solid var(--accent); }
.view { flex: 1; min-height: 0; overflow: auto; position: relative; }
.plot { position: absolute; inset: 0; }
.empty { display: grid; place-items: center; height: 100%; color: var(--muted); padding: var(--s3); }
.muted { color: var(--muted); padding: var(--s1) var(--s2); margin: 0; }

.panel {
  border-left: 1px solid var(--border);
  padding: var(--s2);
  display: flex;
  flex-direction: column;
  gap: var(--s2);
  overflow: auto;
}
.panel h2 { font-size: 13px; font-weight: 600; margin: 0; }
.field { display: flex; flex-direction: column; gap: 4px; }
.field label { color: var(--muted); font-size: 12px; }
.field input, .field select {
  font: inherit;
  color: inherit;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px var(--s1);
  min-width: 0;
}
.field input:focus, .field select:focus { outline: none; border-color: var(--accent); }
.field.invalid input, .field.invalid select { border-color: var(--danger); }
.err { color: var(--danger); font-size: 12px; }
.row { display: flex; gap: var(--s1); }
.row input { flex: 1; }

.btn {
  font: inherit;
  color: var(--text);
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 12px;
  cursor: pointer;
}
.btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.btn:disabled { opacity: 0.5; cursor: default; }
.btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.error { color: var(--danger); white-space: pre-wrap; margin: 0 0 var(--s1); }
details pre { font-size: 11px; overflow: auto; max-height: 240px; text-align: left; }
.banner {
  position: fixed;
  top: var(--s1);
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 12px;
  padding: var(--s1) 12px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
}

table.data { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
table.data th, table.data td {
  text-align: left;
  padding: 6px var(--s2);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
table.data th { color: var(--muted); font-weight: 500; position: sticky; top: 0; background: var(--bg); }
```

- [ ] **Step 6: Implement the rail**

Create `GUI/renderer/components/Rail.tsx`:

```tsx
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
      {stages.map((s) => {
        // reachable when its input exists, or it already has its own result
        const reachable = !s.requires || statusOf[s.requires] === "done" || s.status !== "empty";
        return (
          <button key={s.id} aria-current={s.id === current} disabled={!reachable} onClick={() => onSelect(s.id)}>
            <span class={`dot ${s.status}`} title={s.status} />
            {s.title}
          </button>
        );
      })}
    </nav>
  );
}
```

- [ ] **Step 7: Typecheck**

Run: `cd GUI && npm run typecheck && npm test`
Expected: typecheck exits 0; `4 passed`.

- [ ] **Step 8: Commit**

```bash
git add GUI/renderer
git commit -m "feat(gui): add theme tokens, API client with SSE parsing, and pipeline rail"
```

---

### Task 6: Schema-driven parameter panel

**Files:**
- Create: `GUI/renderer/components/ParamPanel.tsx`
- Test: `GUI/renderer/components/ParamPanel.test.tsx`

**Interfaces:**
- Consumes: `JsonSchema`, `FieldSchema` from `api.ts`; `window.metbit.openFile` from Task 4.
- Produces:
  - `type Values = Record<string, unknown>`, `type FieldErrors = Record<string, string>` (key `""` = form-level error).
  - `defaults(schema: JsonSchema): Values`
  - `fieldErrors(detail: unknown): FieldErrors` (maps a 422 `detail` list `[{loc, msg}]` by `loc[0]`; a string becomes `{"": detail}`).
  - `ParamPanel({ title, schema, initial: Values | null, errors: FieldErrors, busy: boolean, onRun: (v: Values) => void })`. Control ids are `f-<field name>`. Mount with `key={stageId}` so switching stages resets the form.

- [ ] **Step 1: Write the failing test**

Create `GUI/renderer/components/ParamPanel.test.tsx`:

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd GUI && npx vitest run renderer/components`
Expected: FAIL, cannot resolve `./ParamPanel`.

- [ ] **Step 3: Implement `ParamPanel.tsx`**

Create `GUI/renderer/components/ParamPanel.tsx`:

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
}

export function ParamPanel({ title, schema, initial, errors, busy, onRun }: Props) {
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
        <Field key={k} name={k} field={f} value={values[k]} error={errors[k]} onChange={(v) => set(k, v)} />
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
  onChange: (v: unknown) => void;
}

function Field({ name, field, value, error, onChange }: FieldProps) {
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
    // ponytail: CSV is the only file input in stage 1; take filters from the schema when Bruker import lands
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
      {error && <span class="err">{error}</span>}
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd GUI && npm test && npm run typecheck`
Expected: `9 passed`; typecheck exits 0.

- [ ] **Step 5: Commit**

```bash
git add GUI/renderer/components
git commit -m "feat(gui): add schema-driven parameter panel"
```

---

### Task 7: Plot view, table, job bar, and the run flow

**Files:**
- Create: `GUI/renderer/components/PlotView.tsx`
- Create: `GUI/renderer/components/DataTable.tsx`
- Create: `GUI/renderer/components/JobBar.tsx`
- Create: `GUI/renderer/App.tsx`
- Modify: `GUI/renderer/main.tsx` (replace whole file)

**Interfaces:**
- Consumes: `Api`, `ApiError`, `StageInfo`, `TablePage`, `SseEvent` from `api.ts`; `Rail`; `ParamPanel`, `fieldErrors`, `FieldErrors`, `Values`; `window.metbit.onBackendExit`, `window.metbit.restartBackend`.
- Produces:
  - `PlotView({ figure: { data: unknown[]; layout: Record<string, any> } })`, renders into `.plot`.
  - `DataTable({ page: TablePage })`, renders `table.data`.
  - `JobBar({ pct: number | null })`, `null` hides the bar.
  - `App()`: the full UI. Tabs = stage views plus `"table"`.

- [ ] **Step 1: Implement the view components**

Create `GUI/renderer/components/PlotView.tsx`:

```tsx
import Plotly from "plotly.js-dist-min";
import { useEffect, useRef } from "preact/hooks";

interface Figure {
  data: unknown[];
  layout: Record<string, any>;
}

export function PlotView({ figure }: { figure: Figure }) {
  const el = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = el.current!;
    return () => Plotly.purge(node);
  }, []);

  useEffect(() => {
    const css = getComputedStyle(document.documentElement);
    // ponytail: only paper, font and background follow the theme; axis grid colors stay plotly_white
    const layout = {
      ...figure.layout,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { ...figure.layout.font, family: css.getPropertyValue("--font"), size: 12, color: css.getPropertyValue("--text").trim() },
    };
    Plotly.react(el.current!, figure.data, layout, { responsive: true, displaylogo: false });
  }, [figure]);

  return <div ref={el} class="plot" />;
}
```

Create `GUI/renderer/components/DataTable.tsx`:

```tsx
import type { TablePage } from "../api";

const fmt = (v: unknown) => (typeof v === "number" && !Number.isInteger(v) ? v.toPrecision(4) : String(v ?? ""));

export function DataTable({ page }: { page: TablePage }) {
  return (
    <>
      <table class="data">
        <thead>
          <tr>
            {page.columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {page.rows.map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td key={j}>{fmt(v)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {page.total > page.rows.length && (
        <p class="muted">
          Showing {page.rows.length} of {page.total}
        </p>
      )}
    </>
  );
}
```

Create `GUI/renderer/components/JobBar.tsx`:

```tsx
export function JobBar({ pct }: { pct: number | null }) {
  return (
    <div class="jobbar" role="progressbar" aria-valuenow={pct == null ? undefined : Math.round(pct * 100)}>
      <div style={{ width: `${(pct ?? 0) * 100}%`, opacity: pct == null ? 0 : 1 }} />
    </div>
  );
}
```

- [ ] **Step 2: Implement `App.tsx`**

Create `GUI/renderer/App.tsx`:

```tsx
import { useEffect, useState } from "preact/hooks";
import { Api, ApiError, type StageInfo, type TablePage } from "./api";
import { DataTable } from "./components/DataTable";
import { JobBar } from "./components/JobBar";
import { ParamPanel, fieldErrors, type FieldErrors, type Values } from "./components/ParamPanel";
import { PlotView } from "./components/PlotView";
import { Rail } from "./components/Rail";

interface Failure {
  message: string;
  traceback?: string;
}

const label = (t: string) => t[0].toUpperCase() + t.slice(1);

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

  const stage = stages.find((s) => s.id === current);
  const hasResult = stage?.status === "done" || stage?.status === "stale";
  const tabs = stage ? [...stage.views, "table"] : [];

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

  if (!stage) return <div class="empty">Starting metbit…</div>;

  let body;
  if (failure) {
    body = (
      <div class="empty">
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
    body = <div class="empty">{stage.status === "running" ? "Running…" : "Set parameters and press Run"}</div>;
  } else if (tab === "table") {
    body = page && <DataTable page={page} />;
  } else {
    body = figure && <PlotView figure={figure} />;
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

- [ ] **Step 3: Wire the entry point**

Replace `GUI/renderer/main.tsx` with:

```tsx
import "@fontsource-variable/inter";
import "./theme.css";
import { render } from "preact";
import { App } from "./App";

render(<App />, document.getElementById("app")!);
```

- [ ] **Step 4: Typecheck, unit tests, manual run**

Run: `cd GUI && npm run typecheck && npm test && npm start`
Expected: typecheck exits 0; `9 passed`; the app opens with the rail (Import active, PCA disabled). Manually: Browse to a CSV with a `Group` column and ppm headers, Run, see the sample table and a filled Import dot; PCA becomes enabled; run PCA and see the Scores, Loadings, Variance and Table tabs. Enter `n_components` 2 and `pc_y` 3 and Run: a form-level error appears and nothing runs. Toggle macOS dark mode: colors switch.

- [ ] **Step 5: Commit**

```bash
git add GUI/renderer
git commit -m "feat(gui): add plot view, table, job bar and the run flow"
```

---

### Task 8: End-to-end smoke test

**Files:**
- Create: `GUI/playwright.config.ts`
- Create: `GUI/e2e/smoke.spec.ts`

**Interfaces:**
- Consumes: the built app (`npm run build`), control id `#f-path`, `table.data`, rail buttons named by stage title, `.plot`.
- Produces: `npm run e2e` passes locally with a Python that has metbit, fastapi and uvicorn (`METBIT_PYTHON` overrides the interpreter).

- [ ] **Step 1: Write the test**

Create `GUI/playwright.config.ts`:

```ts
import { defineConfig } from "@playwright/test";

export default defineConfig({ testDir: "e2e", timeout: 60_000 });
```

Create `GUI/e2e/smoke.spec.ts`:

```ts
import { _electron as electron, expect, test } from "@playwright/test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function makeCsv(): string {
  const ppm = Array.from({ length: 40 }, (_, i) => (10 - i * 0.25).toFixed(4));
  const rows = Array.from({ length: 12 }, (_, i) => {
    const group = i < 6 ? "A" : "B";
    const values = ppm.map((_, j) => (Math.sin(i * 7 + j) + (group === "B" && j === 5 ? 3 : 0)).toFixed(5));
    return [`s${i}`, group, ...values].join(",");
  });
  const file = join(mkdtempSync(join(tmpdir(), "metbit-e2e-")), "data.csv");
  writeFileSync(file, [["Sample", "Group", ...ppm].join(","), ...rows].join("\n"));
  return file;
}

test("import a CSV and run PCA", async () => {
  const app = await electron.launch({ args: ["."] });
  const win = await app.firstWindow();

  await win.fill("#f-path", makeCsv());
  await win.getByRole("button", { name: "Run" }).click();
  await expect(win.locator("table.data")).toContainText("s11");

  await win.getByRole("button", { name: "PCA" }).click();
  await win.getByRole("button", { name: "Run" }).click();
  await expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

  await app.close();
});
```

- [ ] **Step 2: Run it**

Run: `cd GUI && npm run e2e`
Expected: `1 passed`. If it fails at launch, check `~/Library/Application Support/metbit-gui/logs/server.log`.

- [ ] **Step 3: Full verification**

Run from repo root:
```bash
python3 -m pytest tests/gui --no-cov -q && (cd GUI && npm run typecheck && npm test && npm run e2e)
```
Expected: `20 passed`, typecheck exits 0, `9 passed`, `1 passed`.

- [ ] **Step 4: Commit**

```bash
git add GUI/playwright.config.ts GUI/e2e
git commit -m "test(gui): add Electron end-to-end smoke test for CSV import and PCA"
```

---

## Deferred to later stage plans

- Stage 2: OPLS-DA, STOCSY, Stats, exports, `.metbit` save/load, autosave and reload after restart.
- Stage 3: Bruker import and Process stage (per-sample progress); schema-driven file filters.
- Stage 4: Normalize and Align.
- Stage 5: PyInstaller sidecar build, electron-builder packaging, CI matrix; the e2e test then runs per OS.
- Plot axis/grid colors in dark mode; table paging beyond the first 100 rows.
