# metbit GUI Phase D (Exports, Projects, Autosave) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users keep their work: export tables and matrices as CSV and plots as PNG / SVG, save and reopen `.metbit` projects, and get their session back automatically after a backend crash.

**Architecture:** The sidecar gains a CSV export route (data stages export the full matrix with metadata, re-importable as a CSV; other stages export their table), a `project.py` module that writes a zip of `params.json` plus Parquet for data-stage outputs (never pickle) and restores by loading data directly and re-running analyses, and an autosave hook that saves after every successful job to a path Electron passes on the command line. Electron adds a save dialog and an image writer; plots are exported in the renderer with `Plotly.toImage` from the original metbit figure. Spec: `GUI/docs/specs/2026-09-29-metbit-gui-design.md` (sections "Project file" and "Error handling").

**Tech Stack:** Same as Phase C, plus `pyarrow` (Parquet) on the Python side.

## Global Constraints

- All GUI code lives under `GUI/`; Python GUI tests under `tests/gui/`.
- Never use the em dash character or the section sign anywhere.
- Commit messages carry no `Co-Authored-By` trailer.
- Keep every source file under 500 lines.
- Project format: zip with `params.json` = `{"format": "metbit-gui-project", "version": 1, "stages": {...}}` and `<stage>/X.parquet`, `<stage>/meta.parquet`, `<stage>/labels.parquet`, optional `<stage>/before.parquet` for data stages. No pickle, ever.
- On load: data stages restore directly; analyses saved as `done` re-run; OPLS-DA with `permutations > 0` re-runs with 0 and loads as `stale`; anything else keeps its params with status `empty`.
- Autosave path: `userData/autosave.metbit`, passed to the sidecar as `--autosave <path>`; saved after every successful job; restored only after a backend restart.
- File paths sent to the sidecar must be absolute with an existing parent folder (422 otherwise).
- Test seams (Playwright cannot drive native dialogs): `METBIT_E2E_DIR` makes save dialogs return `<dir>/<default name>`; `METBIT_E2E_OPEN` makes open dialogs return that path; `METBIT_USER_DATA` overrides Electron's userData folder.
- Python GUI tests: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`. Renderer: `cd GUI && npm run typecheck && npm test`. E2E: `cd GUI && npm run e2e`.

## File Map

```
GUI/server/
  requirements.txt                 MODIFY: + pyarrow
  metbit_server/stages/base.py     MODIFY: Stage.export, matrix_frame()
  metbit_server/stages/data.py     MODIFY: exporters on data stages
  metbit_server/project.py         NEW: save_project, load_project, ProjectError
  metbit_server/jobs.py            MODIFY: on_done hook
  metbit_server/app.py             REPLACE: export, save/load/restore routes, autosave
  metbit_server/__main__.py        MODIFY: --autosave
requirements-dev.txt               MODIFY: + pyarrow
tests/gui/test_gui_export.py       NEW
tests/gui/test_gui_project.py      NEW
tests/gui/test_gui_autosave.py     NEW
GUI/electron/files.ts (+ .test.ts) NEW: dataUrlToBuffer
GUI/electron/main.ts, preload.ts   MODIFY: save dialog, image writer, autosave arg, test seams
GUI/renderer/types.d.ts, api.ts    MODIFY
GUI/renderer/components/Rail.tsx (+ test)  MODIFY: Open / Save project footer
GUI/renderer/App.tsx               REPLACE: export actions, projects, restore after restart
GUI/renderer/theme.css             MODIFY
GUI/e2e/project.spec.ts            NEW
```

---

### Task 1: CSV export of tables and matrices

**Files:**
- Modify: `GUI/server/metbit_server/stages/base.py`, `GUI/server/metbit_server/stages/data.py`
- Modify (replace): `GUI/server/metbit_server/app.py`
- Test: `tests/gui/test_gui_export.py`

**Interfaces:**
- Consumes: Phase C stages and app.
- Produces:
  - `base.Stage.export: Optional[Callable[[Any], pd.DataFrame]] = None` (falls back to `table`).
  - `base.matrix_frame(d: Dataset) -> pd.DataFrame`: metadata columns then ppm columns, one row per sample (re-importable as a CSV).
  - `app.PathBody(path: str)`: absolute path with an existing parent, else 422.
  - `POST /stages/{id}/export` body `{"path"}` -> writes CSV, returns `{"path"}`; 409 without a result; 404 when nothing is exportable.
  - `/stages` items gain `exportable: bool`.
  - `create_app(token, stages=STAGES, autosave: str | None = None)` (autosave used in Task 3).

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_export.py`:

```python
import pandas as pd
from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app


def client_after(csv_path, *runs):
    c = TestClient(create_app(TOKEN))
    for sid, params in (("import", {"path": csv_path}), *runs):
        job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
        assert "event: done" in c.get(f"/jobs/{job}/events", headers=AUTH).text
    return c


def test_normalized_matrix_exports_as_a_reimportable_csv(csv_path, tmp_path):
    c = client_after(csv_path, ("normalize", {"method": "pqn"}))
    out = tmp_path / "normalized.csv"
    assert c.post("/stages/normalize/export", json={"path": str(out)}, headers=AUTH).json() == {"path": str(out)}
    df = pd.read_csv(out)
    assert list(df.columns[:2]) == ["Sample", "Group"]
    assert df.shape == (12, 42)
    client_after(str(out))  # the export imports cleanly as a new dataset


def test_analysis_exports_its_table(csv_path, tmp_path):
    c = client_after(csv_path, ("opls", {"group_a": "A", "group_b": "B"}))
    c.post("/stages/opls/export", json={"path": str(tmp_path / "vip.csv")}, headers=AUTH)
    assert "VIP" in pd.read_csv(tmp_path / "vip.csv").columns


def test_export_errors(csv_path, tmp_path):
    c = client_after(csv_path)
    assert c.post("/stages/pca/export", json={"path": str(tmp_path / "x.csv")}, headers=AUTH).status_code == 409
    assert c.post("/stages/import/export", json={"path": "relative.csv"}, headers=AUTH).status_code == 422
    missing_dir = tmp_path / "nope" / "x.csv"
    assert c.post("/stages/import/export", json={"path": str(missing_dir)}, headers=AUTH).status_code == 422
    c = client_after(csv_path, ("stocsy", {"anchor_ppm": 5.0}))
    assert c.post("/stages/stocsy/export", json={"path": str(tmp_path / "s.csv")}, headers=AUTH).status_code == 404


def test_listing_flags_exportable_stages():
    stages = {s["id"]: s for s in TestClient(create_app(TOKEN)).get("/stages", headers=AUTH).json()}
    assert stages["normalize"]["exportable"] is True
    assert stages["pca"]["exportable"] is True
    assert stages["stocsy"]["exportable"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_export.py --no-cov -q -p no:warnings`
Expected: 4 failed (export route missing, `exportable` missing).

- [ ] **Step 3: Stage exporters**

In `GUI/server/metbit_server/stages/base.py`, add this field at the end of the `Stage` dataclass (after `blocked`):

```python
    # CSV export; defaults to `table` when None
    export: Optional[Callable[[Any], pd.DataFrame]] = None
```

and add this function directly after `not_run_figure`:

```python
def matrix_frame(d: Dataset) -> pd.DataFrame:
    """Metadata then ppm columns, one row per sample: re-importable as a CSV."""
    return pd.concat([d.meta.reset_index(drop=True), d.X.reset_index(drop=True)], axis=1)
```

Then run from the repo root:

```bash
python3 - <<'EOF'
import pathlib
p = pathlib.Path("GUI/server/metbit_server/stages/data.py"); s = p.read_text()
pairs = [
    ("from .base import Dataset, Progress, Stage, not_run_figure, spectra_figure, working",
     "from .base import Dataset, Progress, Stage, matrix_frame, not_run_figure, spectra_figure, working"),
    ("def _import_spectra(",
     "def _import_export(d: Union[Dataset, BrukerSource]) -> pd.DataFrame:\n"
     "    return matrix_frame(d) if isinstance(d, Dataset) else d.meta\n\n\ndef _import_spectra("),
    ('        table=lambda d: d.meta,\n        figures={"spectra": _import_spectra},',
     '        table=lambda d: d.meta,\n        export=_import_export,\n        figures={"spectra": _import_spectra},'),
    ('        figures={"spectra": lambda d: spectra_figure(d.X, d.labels, "Processed spectra")},',
     '        export=matrix_frame,\n        figures={"spectra": lambda d: spectra_figure(d.X, d.labels, "Processed spectra")},'),
    ('        figures=_before_after("Normalized"),', '        export=matrix_frame,\n        figures=_before_after("Normalized"),'),
    ('        figures=_before_after("Aligned"),', '        export=matrix_frame,\n        figures=_before_after("Aligned"),'),
]
for a, b in pairs:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
p.write_text(s)
EOF
```

- [ ] **Step 4: App with export (and the Task 3 hooks already wired)**

Replace `GUI/server/metbit_server/app.py` with:

```python
"""FastAPI sidecar for the metbit desktop GUI."""
import asyncio
import json
import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError, field_validator

from .jobs import Busy, Job, start_job
from .session import Session
from .stages import STAGES, Stage, resolve_schema


class PathBody(BaseModel):
    path: str

    @field_validator("path")
    @classmethod
    def _usable(cls, v: str) -> str:
        p = Path(v)
        if not p.is_absolute():
            raise ValueError("Use an absolute path")
        if not p.parent.is_dir():
            raise ValueError(f"Folder does not exist: {p.parent}")
        return v


def create_app(token: str, stages: list[Stage] = STAGES, autosave: Optional[str] = None) -> FastAPI:
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

    @contextmanager
    def exclusive():
        if not session.lock.acquire(blocking=False):
            raise HTTPException(409, "Another job is running")
        try:
            yield
        finally:
            session.lock.release()

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
                "exportable": (s.export or s.table) is not None,
                "blocked": s.blocked(session) if s.blocked else None,
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
        reason = stage.blocked(session) if stage.blocked else None
        if reason:
            raise HTTPException(409, reason)
        try:
            job = start_job(session, stage, params, on_done=autosave_now if autosave else None)
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
        # double_precision=15 (the max): the default 10 rounds p-values like 3e-11 to 0
        page = json.loads(df.iloc[offset:offset + limit].to_json(orient="split", index=False, double_precision=15))
        return {"columns": [str(c) for c in page["columns"]], "rows": page["data"], "total": len(df)}

    @app.post("/stages/{sid}/export")
    def export(sid: str, body: PathBody):
        stage = stage_or_404(sid)
        make = stage.export or stage.table
        if make is None:
            raise HTTPException(404, f"Stage '{sid}' has nothing to export as CSV")
        make(result_or_409(sid)).to_csv(body.path, index=False)
        return {"path": body.path}

    def autosave_now() -> None:
        # replaced in Task 3 by a real save; kept as a no-op so Task 1 stands alone
        return None

    return app
```

The app passes an `on_done` hook to `start_job`, so replace `GUI/server/metbit_server/jobs.py` with:

```python
"""Run one stage at a time on a worker thread and queue its events for SSE."""
import logging
import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from pydantic import BaseModel

log = logging.getLogger(__name__)


class Busy(Exception):
    """Another job holds the session lock."""


@dataclass
class Job:
    id: str
    events: "queue.Queue[tuple[str, dict]]" = field(default_factory=queue.Queue)


def start_job(session, stage, params: BaseModel, on_done: Optional[Callable[[], None]] = None) -> Job:
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
            if on_done:
                try:
                    on_done()  # still under the lock, so the saved state is consistent
                except Exception:
                    log.exception("autosave failed")
        except Exception as e:
            session.fail(stage.id)
            event = ("error", {"message": str(e), "traceback": traceback.format_exc()})
        # release before announcing, so a client reacting to "done" can run again at once
        session.lock.release()
        job.events.put(event)

    threading.Thread(target=work, daemon=True).start()
    return job
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `82 passed` (78 + 4; 80 passed and 2 skipped without MTBLS2052).

- [ ] **Step 6: Commit**

```bash
git add GUI/server tests/gui
git commit -m "feat(gui): export stage tables and matrices as CSV"
```

---

### Task 2: `.metbit` project save and load

**Files:**
- Create: `GUI/server/metbit_server/project.py`
- Modify: `GUI/server/requirements.txt`, `requirements-dev.txt` (add `pyarrow`)
- Test: `tests/gui/test_gui_project.py`

**Interfaces:**
- Consumes: `Session` (`order`, `after`, `status`, `params`, `outputs`), `STAGES`, `DATA_ORDER`, `Dataset`, `BrukerSource`.
- Produces:
  - `project.ProjectError(ValueError)`.
  - `project.save_project(session: Session, path: str) -> None`.
  - `project.load_project(session: Session, stages: list[Stage], path: str) -> None`; replaces the session's `status` / `params` / `outputs` only when the whole file loads; raises `ProjectError("Not a metbit project: ...")` / `ProjectError("This project was saved by a newer metbit GUI")`.

- [ ] **Step 1: Dependency**

Append `pyarrow` as its own line to `GUI/server/requirements.txt`, and add `pyarrow` under the `# GUI sidecar tests (tests/gui)` block in `requirements-dev.txt`.

Run: `python3 -c "import pyarrow; print(pyarrow.__version__)"`
Expected: a version number (install with `python3 -m pip install pyarrow` otherwise).

- [ ] **Step 2: Write the failing tests**

Create `tests/gui/test_gui_project.py`:

```python
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from metbit_server.project import ProjectError, load_project, save_project
from metbit_server.session import Session
from metbit_server.stages import STAGES, BrukerSource


def fresh() -> Session:
    return Session([s.id for s in STAGES], {s.id: s.after for s in STAGES})


def test_roundtrip_restores_data_and_reruns_analyses(run_all, tmp_path):
    s = run_all(
        ("normalize", {"method": "pqn"}),
        ("pca", {"n_components": 3}),
        ("opls", {"group_a": "A", "group_b": "B", "permutations": 5}),
        ("stocsy", {"anchor_ppm": 5.0}),
    )
    s.status["stocsy"] = "stale"  # as if its input changed after it ran
    path = tmp_path / "study.metbit"
    save_project(s, str(path))

    t = fresh()
    load_project(t, STAGES, str(path))
    assert (t.status["import"], t.status["normalize"], t.status["pca"]) == ("done", "done", "done")
    assert t.output("normalize").X.equals(s.output("normalize").X)
    assert t.output("normalize").before.equals(s.output("normalize").before)
    assert t.output("import").meta.equals(s.output("import").meta)
    assert list(t.output("import").labels) == list(s.output("import").labels)
    scores = lambda sess: sess.output("pca").model.get_scores().select_dtypes("number").to_numpy()
    assert np.allclose(scores(t), scores(s))
    # the slow permutation test is not repeated on load
    assert t.status["opls"] == "stale" and not t.output("opls").permuted
    assert t.params["opls"]["permutations"] == 5
    # stale analyses keep their params but no result
    assert t.status["stocsy"] == "empty" and t.params["stocsy"]["anchor_ppm"] == 5.0
    assert t.status["volcano"] == "empty" and "volcano" not in t.params


def test_project_is_plain_json_and_parquet(run_all, tmp_path):
    save_project(run_all(), str(tmp_path / "p.metbit"))
    with zipfile.ZipFile(tmp_path / "p.metbit") as zf:
        assert sorted(zf.namelist()) == ["import/X.parquet", "import/labels.parquet", "import/meta.parquet", "params.json"]
        manifest = json.loads(zf.read("params.json"))
    assert (manifest["format"], manifest["version"]) == ("metbit-gui-project", 1)
    assert manifest["stages"]["import"]["status"] == "done"


def test_bruker_import_roundtrip(run_all, tmp_path):
    for n in ["61020601", "61030601"]:
        (tmp_path / "exps" / n).mkdir(parents=True)
        (tmp_path / "exps" / n / "fid").write_bytes(b"")
    pd.DataFrame({"fid": [61020601, 61030601], "Group": ["sham", "nx"]}).to_csv(tmp_path / "map.csv", index=False)
    s = run_all(("import", {
        "source": "bruker", "folder": str(tmp_path / "exps"),
        "metadata_path": str(tmp_path / "map.csv"), "sample_column": "fid",
    }))
    save_project(s, str(tmp_path / "b.metbit"))
    t = fresh()
    load_project(t, STAGES, str(tmp_path / "b.metbit"))
    src = t.output("import")
    assert isinstance(src, BrukerSource)
    assert src.experiments == s.output("import").experiments
    assert list(src.labels) == ["sham", "nx"]


def test_rejects_files_that_are_not_projects(tmp_path):
    bad = tmp_path / "text.metbit"
    bad.write_text("nope")
    other = tmp_path / "other.metbit"
    with zipfile.ZipFile(other, "w") as zf:
        zf.writestr("params.json", json.dumps({"format": "something-else"}))
    newer = tmp_path / "newer.metbit"
    with zipfile.ZipFile(newer, "w") as zf:
        zf.writestr("params.json", json.dumps({"format": "metbit-gui-project", "version": 99, "stages": {}}))
    t = fresh()
    for path, msg in [(bad, "Not a metbit project"), (other, "Not a metbit project"),
                      (tmp_path / "missing.metbit", "Not a metbit project"), (newer, "newer metbit GUI")]:
        with pytest.raises(ProjectError, match=msg):
            load_project(t, STAGES, str(path))
    assert set(t.status.values()) == {"empty"}  # a failed load leaves the session untouched
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_project.py --no-cov -q -p no:warnings`
Expected: FAIL with `ModuleNotFoundError: No module named 'metbit_server.project'`.

- [ ] **Step 4: Implement**

Create `GUI/server/metbit_server/project.py`:

```python
"""Save and load a GUI session as a .metbit project: a zip of JSON and Parquet, never pickle.

Fitted models are not stored (pickled objects would run code when a shared project is opened).
Data stages restore directly; analyses re-run from their saved params.
"""
import io
import json
import logging
import zipfile

import pandas as pd

from .session import Session
from .stages import DATA_ORDER, BrukerSource, Dataset, Stage

FORMAT = "metbit-gui-project"
VERSION = 1
log = logging.getLogger(__name__)


class ProjectError(ValueError):
    pass


def _write(zf: zipfile.ZipFile, name: str, df: pd.DataFrame) -> None:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]  # Parquet wants string column names
    buf = io.BytesIO()
    out.to_parquet(buf)
    zf.writestr(name, buf.getvalue())


def _read(zf: zipfile.ZipFile, name: str, ppm_columns: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(io.BytesIO(zf.read(name)))
    if ppm_columns:
        df.columns = df.columns.astype(float)
    return df


def save_project(session: Session, path: str) -> None:
    manifest: dict = {"format": FORMAT, "version": VERSION, "stages": {}}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for sid in session.order:
            if sid not in session.params:
                continue
            entry = {"status": session.status[sid], "params": session.params[sid]}
            out = session.outputs.get(sid)
            if sid in DATA_ORDER and session.status[sid] in ("done", "stale"):
                if isinstance(out, (Dataset, BrukerSource)):
                    _write(zf, f"{sid}/meta.parquet", out.meta)
                    _write(zf, f"{sid}/labels.parquet", out.labels.to_frame("label"))
                if isinstance(out, Dataset):
                    entry["data"] = "dataset"
                    _write(zf, f"{sid}/X.parquet", out.X)
                    if out.before is not None:
                        _write(zf, f"{sid}/before.parquet", out.before)
                elif isinstance(out, BrukerSource):
                    entry["data"] = "bruker"
                    entry["experiments"] = [list(e) for e in out.experiments]
            manifest["stages"][sid] = entry
        zf.writestr("params.json", json.dumps(manifest, indent=2))


def _open(path: str) -> tuple[zipfile.ZipFile, dict]:
    try:
        zf = zipfile.ZipFile(path)
        manifest = json.loads(zf.read("params.json"))
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as e:
        raise ProjectError(f"Not a metbit project: {e}") from e
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ProjectError("Not a metbit project: unknown format")
    if manifest.get("version", 0) > VERSION:
        raise ProjectError("This project was saved by a newer metbit GUI")
    return zf, manifest


def load_project(session: Session, stages: list[Stage], path: str) -> None:
    zf, manifest = _open(path)
    saved = manifest.get("stages", {})
    by_id = {s.id: s for s in stages}
    fresh = Session(session.order, session.after)  # swapped in only after everything loads
    with zf:
        for sid in session.order:
            entry = saved.get(sid)
            if not entry or sid not in by_id:
                continue
            fresh.params[sid] = by_id[sid].params.model_validate(entry["params"]).model_dump()
            kind = entry.get("data")
            if kind not in ("dataset", "bruker"):
                continue
            meta = _read(zf, f"{sid}/meta.parquet")
            labels = _read(zf, f"{sid}/labels.parquet")["label"].astype(str).reset_index(drop=True)
            if kind == "dataset":
                before_name = f"{sid}/before.parquet"
                before = _read(zf, before_name, ppm_columns=True) if before_name in zf.namelist() else None
                fresh.outputs[sid] = Dataset(
                    X=_read(zf, f"{sid}/X.parquet", ppm_columns=True), meta=meta, labels=labels, before=before
                )
            else:
                experiments = [tuple(e) for e in entry["experiments"]]
                fresh.outputs[sid] = BrukerSource(experiments=experiments, meta=meta, labels=labels)
            fresh.status[sid] = entry["status"]

    for stage in stages:
        entry = saved.get(stage.id)
        if stage.id in DATA_ORDER or not entry or entry.get("status") != "done":
            continue
        params = stage.params.model_validate(entry["params"])
        slow = getattr(params, "permutations", 0) > 0
        if slow:  # OPLS-DA permutation tests can take minutes; the user re-runs them on demand
            params = params.model_copy(update={"permutations": 0})
        try:
            fresh.outputs[stage.id] = stage.run(params, fresh, lambda _pct: None)
        except Exception:
            log.exception("could not re-run %s while loading %s", stage.id, path)
            continue
        fresh.status[stage.id] = "stale" if slow else "done"

    session.status, session.params, session.outputs = fresh.status, fresh.params, fresh.outputs
```

- [ ] **Step 5: Run tests to verify they pass**


Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `86 passed` (84 passed and 2 skipped without MTBLS2052).

- [ ] **Step 6: Commit**

```bash
git add GUI/server requirements-dev.txt tests/gui
git commit -m "feat(gui): save and load .metbit projects as JSON plus Parquet"
```

---

### Task 3: Project routes, autosave and restore

**Files:**
- Modify: `GUI/server/metbit_server/app.py`, `GUI/server/metbit_server/__main__.py`
- Test: `tests/gui/test_gui_autosave.py`

**Interfaces:**
- Consumes: `save_project`, `load_project`, `ProjectError` from Task 2; `create_app(..., autosave=...)`, `exclusive()`, `PathBody` from Task 1.
- Produces:
  - `POST /session/save` `{"path"}` -> `{"path"}`; `POST /session/load` `{"path"}` -> `{"path"}` (422 with the `ProjectError` message on a bad file); `POST /session/restore` -> `{"restored": bool}` (loads the autosave file when one exists).
  - Autosave after every successful job when `autosave` is set; a failed job does not autosave.
  - CLI: `python -m metbit_server --port <p> [--autosave <path>]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_autosave.py`:

```python
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app

SERVER_DIR = Path(__file__).resolve().parents[2] / "GUI" / "server"


def run(c, sid, params):
    job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
    return c.get(f"/jobs/{job}/events", headers=AUTH).text


def statuses(c):
    return {s["id"]: s["status"] for s in c.get("/stages", headers=AUTH).json()}


def test_autosave_after_each_job_then_restore(csv_path, tmp_path):
    auto = tmp_path / "autosave.metbit"
    c = TestClient(create_app(TOKEN, autosave=str(auto)))
    run(c, "import", {"path": csv_path})
    assert auto.is_file()
    run(c, "pca", {})
    restarted = TestClient(create_app(TOKEN, autosave=str(auto)))
    assert restarted.post("/session/restore", headers=AUTH).json() == {"restored": True}
    st = statuses(restarted)
    assert (st["import"], st["pca"]) == ("done", "done")


def test_restore_without_an_autosave(tmp_path):
    assert TestClient(create_app(TOKEN)).post("/session/restore", headers=AUTH).json() == {"restored": False}
    c = TestClient(create_app(TOKEN, autosave=str(tmp_path / "none.metbit")))
    assert c.post("/session/restore", headers=AUTH).json() == {"restored": False}


def test_failed_job_does_not_autosave(tmp_path):
    auto = tmp_path / "autosave.metbit"
    c = TestClient(create_app(TOKEN, autosave=str(auto)))
    assert "event: error" in run(c, "import", {"path": str(tmp_path / "missing.csv")})
    assert not auto.exists()


def test_save_and_load_routes(csv_path, tmp_path):
    c = TestClient(create_app(TOKEN))
    run(c, "import", {"path": csv_path})
    target = tmp_path / "study.metbit"
    assert c.post("/session/save", json={"path": str(target)}, headers=AUTH).json() == {"path": str(target)}
    other = TestClient(create_app(TOKEN))
    assert other.post("/session/load", json={"path": str(target)}, headers=AUTH).status_code == 200
    assert statuses(other)["import"] == "done"
    (tmp_path / "junk.metbit").write_text("junk")
    r = other.post("/session/load", json={"path": str(tmp_path / "junk.metbit")}, headers=AUTH)
    assert r.status_code == 422 and "Not a metbit project" in r.json()["detail"]
    assert c.post("/session/save", json={"path": "relative.metbit"}, headers=AUTH).status_code == 422


def test_cli_accepts_autosave():
    env = {**os.environ, "PYTHONPATH": str(SERVER_DIR.parents[1])}  # repo root, so `import metbit` works
    out = subprocess.run(
        [sys.executable, "-m", "metbit_server", "--help"], cwd=SERVER_DIR, env=env, capture_output=True, text=True, check=True
    )
    assert "--autosave" in out.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_autosave.py --no-cov -q -p no:warnings`
Expected: 5 failed (routes return 404/405, no autosave file, no `--autosave` flag).

- [ ] **Step 3: Routes and the real autosave**

In `GUI/server/metbit_server/app.py`, add after the `from .jobs import ...` line:

```python
from .project import ProjectError, load_project, save_project
```

and replace the placeholder block

```python
    def autosave_now() -> None:
        # replaced in Task 3 by a real save; kept as a no-op so Task 1 stands alone
        return None
```

with:

```python
    def autosave_now() -> None:
        save_project(session, autosave)

    @app.post("/session/save")
    def save(body: PathBody):
        with exclusive():
            save_project(session, body.path)
        return {"path": body.path}

    @app.post("/session/load")
    def load(body: PathBody):
        with exclusive():
            try:
                load_project(session, stages, body.path)
            except ProjectError as e:
                raise HTTPException(422, str(e))
        return {"path": body.path}

    @app.post("/session/restore")
    def restore():
        if not autosave or not Path(autosave).is_file():
            return {"restored": False}
        with exclusive():
            try:
                load_project(session, stages, autosave)
            except ProjectError:
                return {"restored": False}
        return {"restored": True}
```

- [ ] **Step 4: CLI flag**

Replace `GUI/server/metbit_server/__main__.py` with:

```python
"""Entry point: METBIT_TOKEN=<token> python -m metbit_server --port <port> [--autosave <path>]"""
import argparse
import os
import sys

import uvicorn

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="metbit_server")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--autosave", help="project file written after every successful job")
    args = parser.parse_args()
    token = os.environ.get("METBIT_TOKEN")
    if not token:
        sys.exit("METBIT_TOKEN is not set")
    uvicorn.run(create_app(token, autosave=args.autosave), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**


Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `91 passed` (89 passed and 2 skipped without MTBLS2052).

- [ ] **Step 6: Commit**

```bash
git add GUI/server tests/gui
git commit -m "feat(gui): project save, load and autosave routes with restore"
```

---

### Task 4: Electron: save dialogs, image writer, autosave path, test seams

**Files:**
- Create: `GUI/electron/files.ts`, `GUI/electron/files.test.ts`
- Modify: `GUI/electron/main.ts`, `GUI/electron/preload.ts`, `GUI/renderer/types.d.ts`

**Interfaces:**
- Consumes: sidecar CLI `--autosave`.
- Produces:
  - `files.dataUrlToBuffer(url: string): Buffer` (base64 and URI-encoded data URLs; throws `Not a data URL`).
  - `window.metbit.saveDialog(defaultName, filters?) -> Promise<string | null>`; `window.metbit.saveImage(defaultName, dataUrl) -> Promise<string | null>` (asks for a path, writes the image, returns the path).
  - Env seams `METBIT_E2E_DIR`, `METBIT_E2E_OPEN`, `METBIT_USER_DATA` (see Global Constraints).

- [ ] **Step 1: Write the failing test**

Create `GUI/electron/files.test.ts`:

```ts
// @vitest-environment node
import { expect, it } from "vitest";
import { dataUrlToBuffer } from "./files";

it("decodes base64 PNG data URLs", () => {
  const png = Buffer.from([0x89, 0x50, 0x4e, 0x47]);
  expect(dataUrlToBuffer(`data:image/png;base64,${png.toString("base64")}`)).toEqual(png);
});

it("decodes URI-encoded SVG data URLs", () => {
  const svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>δ 3.05</text></svg>';
  expect(dataUrlToBuffer(`data:image/svg+xml,${encodeURIComponent(svg)}`).toString("utf8")).toBe(svg);
});

it("rejects anything else", () => {
  expect(() => dataUrlToBuffer("/etc/passwd")).toThrow("Not a data URL");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd GUI && npx vitest run electron/files.test.ts`
Expected: FAIL, cannot resolve `./files`.

- [ ] **Step 3: Implement**

Create `GUI/electron/files.ts`:

```ts
export function dataUrlToBuffer(url: string): Buffer {
  const comma = url.indexOf(",");
  if (!url.startsWith("data:") || comma < 0) throw new Error("Not a data URL");
  const header = url.slice(5, comma);
  const body = url.slice(comma + 1);
  return header.endsWith(";base64") ? Buffer.from(body, "base64") : Buffer.from(decodeURIComponent(body), "utf8");
}
```

Run: `cd GUI && npx vitest run electron/files.test.ts`
Expected: `3 passed`.

- [ ] **Step 4: Main process**

In `GUI/electron/main.ts`:

Replace the import block at the top with:

```ts
import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { writeFile } from "node:fs/promises";
import { extname, join } from "node:path";
import { dataUrlToBuffer } from "./files";
import { startSidecar, type Backend, type SidecarCommand } from "./sidecar";

// test seams: Playwright cannot drive native dialogs, and tests keep their own userData
const E2E_DIR = process.env.METBIT_E2E_DIR;
const E2E_OPEN = process.env.METBIT_E2E_OPEN;
if (process.env.METBIT_USER_DATA) app.setPath("userData", process.env.METBIT_USER_DATA);
```

In `sidecarCommand()`, pass the autosave path in both branches: change `args: []` to `args: ["--autosave", autosavePath()]` and `args: ["-m", "metbit_server"]` to `args: ["-m", "metbit_server", "--autosave", autosavePath()]`, and add above `sidecarCommand`:

```ts
const autosavePath = () => join(app.getPath("userData"), "autosave.metbit");
```

Replace the `dialog:openFile` handler with:

```ts
ipcMain.handle("dialog:openFile", async (_e, filters?: Electron.FileFilter[], directory?: boolean) => {
  if (E2E_OPEN) return E2E_OPEN;
  const r = await dialog.showOpenDialog(win!, { properties: [directory ? "openDirectory" : "openFile"], filters });
  return r.canceled ? null : r.filePaths[0];
});

async function chooseSavePath(defaultName: string, filters?: Electron.FileFilter[]): Promise<string | null> {
  if (E2E_DIR) return join(E2E_DIR, defaultName);
  const r = await dialog.showSaveDialog(win!, { defaultPath: defaultName, filters });
  return r.canceled || !r.filePath ? null : r.filePath;
}

ipcMain.handle("dialog:save", (_e, defaultName: string, filters?: Electron.FileFilter[]) => chooseSavePath(defaultName, filters));
ipcMain.handle("file:saveImage", async (_e, defaultName: string, dataUrl: string) => {
  const path = await chooseSavePath(defaultName, [{ name: "Image", extensions: [extname(defaultName).slice(1)] }]);
  if (path) await writeFile(path, dataUrlToBuffer(dataUrl));
  return path;
});
```

In `GUI/electron/preload.ts`, add inside the exposed object:

```ts
  saveDialog: (defaultName: string, filters?: { name: string; extensions: string[] }[]) =>
    ipcRenderer.invoke("dialog:save", defaultName, filters),
  saveImage: (defaultName: string, dataUrl: string) => ipcRenderer.invoke("file:saveImage", defaultName, dataUrl),
```

In `GUI/renderer/types.d.ts`, add inside `metbit: { ... }`:

```ts
    saveDialog(defaultName: string, filters?: { name: string; extensions: string[] }[]): Promise<string | null>;
    saveImage(defaultName: string, dataUrl: string): Promise<string | null>;
```

- [ ] **Step 5: Verify**

Run: `cd GUI && npm run typecheck && npm test`
Expected: typecheck exits 0; `20 passed`.

- [ ] **Step 6: Commit**

```bash
git add GUI/electron GUI/renderer/types.d.ts
git commit -m "feat(gui): save dialogs, image writer and autosave path in Electron"
```

---

### Task 5: Renderer: export actions, projects, restore after restart

**Files:**
- Modify: `GUI/renderer/api.ts`, `GUI/renderer/components/Rail.tsx`, `GUI/renderer/components/Rail.test.tsx`, `GUI/renderer/theme.css`
- Modify (replace): `GUI/renderer/App.tsx`

**Interfaces:**
- Consumes: `/stages` field `exportable`; routes from Tasks 1 and 3; `window.metbit.saveDialog`, `saveImage`, `openFile`.
- Produces:
  - `StageInfo.exportable: boolean`; `Api.exportCsv(id, path)`, `Api.saveProject(path)`, `Api.loadProject(path)`, `Api.restore(): Promise<{ restored: boolean }>`.
  - `Rail` optional props `onOpen`, `onSave` render a `.rail-footer` with buttons `Open project…` / `Save project…`.
  - App tab bar actions: `Export CSV` (when `exportable`), `PNG`, `SVG` (on plot views); a `.notice` message for 4 s after a save.

- [ ] **Step 1: Write the failing test**

In `GUI/renderer/components/Rail.test.tsx`, add `exportable: true,` after `blocked: null,` in the `stage` helper, then append:

```tsx
it("offers project open and save in the footer", () => {
  const root = document.createElement("div");
  const calls: string[] = [];
  render(
    <Rail stages={[stage("import", "Data")]} current="import" onSelect={() => {}} onOpen={() => calls.push("open")} onSave={() => calls.push("save")} />,
    root,
  );
  const button = (text: string) => [...root.querySelectorAll(".rail-footer button")].find((b) => b.textContent === text) as HTMLButtonElement;
  button("Open project…").click();
  button("Save project…").click();
  expect(calls).toEqual(["open", "save"]);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd GUI && npm test`
Expected: 1 failed (no `.rail-footer`).

- [ ] **Step 3: API and Rail**

In `GUI/renderer/api.ts`, add `exportable: boolean;` after `has_table: boolean;` in `StageInfo`, and add these methods to `class Api` after `table(...)`:

```ts
  private post(path: string, body?: unknown): Promise<any> {
    return this.req(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) }).then((r) => r.json());
  }

  exportCsv(id: string, path: string): Promise<unknown> {
    return this.post(`/stages/${id}/export`, { path });
  }

  saveProject(path: string): Promise<unknown> {
    return this.post("/session/save", { path });
  }

  loadProject(path: string): Promise<unknown> {
    return this.post("/session/load", { path });
  }

  restore(): Promise<{ restored: boolean }> {
    return this.post("/session/restore");
  }
```

Replace `GUI/renderer/components/Rail.tsx` with:

```tsx
import { Fragment } from "preact";
import type { StageInfo } from "../api";

interface Props {
  stages: StageInfo[];
  current: string;
  onSelect: (id: string) => void;
  onOpen?: () => void;
  onSave?: () => void;
}

export function Rail({ stages, current, onSelect, onOpen, onSave }: Props) {
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
      {(onOpen || onSave) && (
        <div class="rail-footer">
          {onOpen && <button onClick={onOpen}>Open project…</button>}
          {onSave && <button onClick={onSave}>Save project…</button>}
        </div>
      )}
    </nav>
  );
}
```

Append to `GUI/renderer/theme.css`:

```css
.rail-footer { margin-top: auto; border-top: 1px solid var(--border); padding-top: var(--s1); display: flex; flex-direction: column; gap: 2px; }
.tabs-right { margin-left: auto; display: flex; align-items: center; gap: 12px; }
.link { all: unset; cursor: pointer; color: var(--muted); font-size: 12px; }
.link:hover { color: var(--text); }
.link:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.notice { color: var(--accent); font-size: 12px; }
```

- [ ] **Step 4: App**

Replace `GUI/renderer/App.tsx` with:

```tsx
import Plotly from "plotly.js-dist-min";
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

const LABELS: Record<string, string> = { vip: "VIP", "s-plot": "S-plot", predicted: "Predicted vs actual" };
const label = (t: string) => LABELS[t] ?? t[0].toUpperCase() + t.slice(1);
const basename = (p: string) => p.split(/[\\/]/).pop() ?? p;
const PROJECT = [{ name: "metbit project", extensions: ["metbit"] }];
const IMAGE = { width: 1600, height: 1000 };

export function App() {
  const [api, setApi] = useState<Api | null>(null);
  const [stages, setStages] = useState<StageInfo[]>([]);
  const [current, setCurrent] = useState("import");
  const [tab, setTab] = useState("table");
  const [version, setVersion] = useState(0); // bumps after every run to refetch views
  const [formKey, setFormKey] = useState(0); // bumps when a project replaces the params
  const [figure, setFigure] = useState<any>(null);
  const [page, setPage] = useState<TablePage | null>(null);
  const [pct, setPct] = useState<number | null>(null);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [failure, setFailure] = useState<Failure | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  const [picked, setPicked] = useState<number | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

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

  function flash(message: string) {
    setNotice(message);
    setTimeout(() => setNotice((n) => (n === message ? null : n)), 4000);
  }

  function select(id: string) {
    setCurrent(id);
    setErrors({});
    setFailure(null);
    setTab(stages.find((s) => s.id === id)?.views[0] ?? "table");
  }

  async function refresh(client: Api) {
    setStages(await client.stages());
    setVersion((v) => v + 1);
    setFormKey((k) => k + 1);
  }

  async function guarded(action: () => Promise<void>) {
    try {
      await action();
    } catch (e) {
      setFailure({ message: (e as Error).message });
    }
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

  const exportCsv = () =>
    guarded(async () => {
      const path = await window.metbit.saveDialog(`${current}.csv`, [{ name: "CSV", extensions: ["csv"] }]);
      if (!path || !api) return;
      await api.exportCsv(current, path);
      flash(`Saved ${basename(path)}`);
    });

  const exportImage = (format: "png" | "svg") =>
    guarded(async () => {
      if (!figure) return;
      // the original metbit figure (white background), not the themed on-screen one
      const url = await Plotly.toImage({ data: figure.data, layout: { ...figure.layout, ...IMAGE } }, { format, ...IMAGE });
      const path = await window.metbit.saveImage(`${current}-${tab}.${format}`, url);
      if (path) flash(`Saved ${basename(path)}`);
    });

  const saveProject = () =>
    guarded(async () => {
      const path = await window.metbit.saveDialog("analysis.metbit", PROJECT);
      if (!path || !api) return;
      await api.saveProject(path);
      flash(`Saved ${basename(path)}`);
    });

  const openProject = () =>
    guarded(async () => {
      const path = await window.metbit.openFile(PROJECT);
      if (!path || !api) return;
      await api.loadProject(path);
      await refresh(api);
      select("import");
      flash(`Opened ${basename(path)}`);
    });

  async function restart() {
    await window.metbit.restartBackend();
    setBackendDown(false);
    const next = await Api.connect();
    const { restored } = await next.restore(); // before setApi, so the stage list reflects it
    setApi(next);
    setFormKey((k) => k + 1);
    select("import");
    if (restored) flash("Session restored");
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
      <Rail stages={stages} current={current} onSelect={select} onOpen={openProject} onSave={saveProject} />
      <main class="center">
        <div class="tabs" role="tablist">
          {hasResult &&
            tabs.map((t) => (
              <button key={t} role="tab" aria-selected={t === tab} onClick={() => setTab(t)}>
                {label(t)}
              </button>
            ))}
          <div class="tabs-right">
            {notice && (
              <span class="notice" aria-live="polite">
                {notice}
              </span>
            )}
            {picked !== null && <span class="picked">δ {picked.toFixed(3)} ppm</span>}
            {hasResult && stage.exportable && (
              <button class="link" onClick={exportCsv}>
                Export CSV
              </button>
            )}
            {hasResult && tab !== "table" && figure && (
              <>
                <button class="link" onClick={() => exportImage("png")}>
                  PNG
                </button>
                <button class="link" onClick={() => exportImage("svg")}>
                  SVG
                </button>
              </>
            )}
          </div>
        </div>
        <div class="view">{body}</div>
      </main>
      <ParamPanel
        key={`${current}:${formKey}`}
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

- [ ] **Step 5: Verify**

Run: `cd GUI && npm run typecheck && npm test && npm run build`
Expected: typecheck exits 0; `21 passed`; build succeeds.

- [ ] **Step 6: Commit**

```bash
git add GUI/renderer
git commit -m "feat(gui): export actions, project open and save, restore after restart"
```

---

### Task 6: End-to-end exports, projects and crash recovery

**Files:**
- Create: `GUI/e2e/project.spec.ts`

**Interfaces:**
- Consumes: env seams from Task 4; buttons `Export CSV`, `PNG`, `Save project…`, `Open project…`, `Restart`.
- Produces: `npm run e2e` runs 6 tests.

- [ ] **Step 1: Write the tests**

Create `GUI/e2e/project.spec.ts`:

```ts
import { _electron as electron, expect, test, type Page } from "@playwright/test";
import { execSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function makeCsv(dir: string): string {
  const ppm = Array.from({ length: 40 }, (_, i) => (10 - i * 0.25).toFixed(4));
  const rows = Array.from({ length: 12 }, (_, i) => {
    const group = i < 6 ? "A" : "B";
    const values = ppm.map((_, j) => (Math.abs(Math.sin(i * 7 + j)) + 1 + (group === "B" && j === 5 ? 3 : 0)).toFixed(5));
    return [`s${i}`, group, ...values].join(",");
  });
  const file = join(dir, "data.csv");
  writeFileSync(file, [["Sample", "Group", ...ppm].join(","), ...rows].join("\n"));
  return file;
}

const run = (win: Page) => win.getByRole("button", { name: "Run", exact: true }).click();
const done = (win: Page, name: string) => expect(win.locator(".rail button", { hasText: name }).locator(".dot.done")).toBeVisible();
const plot = (win: Page) => expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

async function launch(env: Record<string, string>) {
  const app = await electron.launch({ args: ["."], env: { ...process.env, ...env } as Record<string, string> });
  return { app, win: await app.firstWindow() };
}

async function importAndPca(win: Page, csv: string) {
  await win.fill("#f-path", csv);
  await run(win);
  await expect(win.locator("table.data")).toContainText("s11");
  await win.locator(".rail button", { hasText: "PCA" }).click();
  await run(win);
  await plot(win);
}

test("export CSV and PNG, save a project and reopen it", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const { app, win } = await launch({ METBIT_E2E_DIR: dir, METBIT_USER_DATA: join(dir, "user") });
  await importAndPca(win, makeCsv(dir));

  await win.getByRole("button", { name: "Export CSV" }).click();
  await expect.poll(() => existsSync(join(dir, "pca.csv"))).toBe(true);
  await win.getByRole("button", { name: "PNG" }).click();
  await expect.poll(() => existsSync(join(dir, "pca-scores.png"))).toBe(true);
  expect(readFileSync(join(dir, "pca-scores.png")).subarray(1, 4).toString()).toBe("PNG");
  await win.getByRole("button", { name: "Save project…" }).click();
  await expect.poll(() => existsSync(join(dir, "analysis.metbit"))).toBe(true);
  await app.close();

  const again = await launch({ METBIT_E2E_OPEN: join(dir, "analysis.metbit"), METBIT_USER_DATA: join(dir, "user2") });
  await again.win.getByRole("button", { name: "Open project…" }).click();
  await done(again.win, "PCA");
  await again.win.locator(".rail button", { hasText: "PCA" }).click();
  await plot(again.win);
  await again.app.close();
});

test("a backend crash restores the session after Restart", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const { app, win } = await launch({ METBIT_USER_DATA: join(dir, "user") });
  await importAndPca(win, makeCsv(dir));

  execSync("pkill -f metbit_server");
  await expect(win.getByRole("alert")).toContainText("Backend stopped");
  await win.getByRole("button", { name: "Restart" }).click();
  await expect(win.getByText("Session restored")).toBeVisible({ timeout: 30_000 });
  await done(win, "PCA");
  await app.close();
});
```

- [ ] **Step 2: Run everything**

Run from the repo root:
```bash
python3 -m pytest -q -p no:cacheprovider && (cd GUI && npm run typecheck && npm test && npm run e2e)
```
Expected: repo suite passes with coverage at or above 99%; typecheck exits 0; `21 passed`; `6 passed` (5 plus 1 skipped without MTBLS2052).

- [ ] **Step 3: Commit**

```bash
git add GUI/e2e/project.spec.ts
git commit -m "test(gui): cover exports, projects and crash recovery end to end"
```
