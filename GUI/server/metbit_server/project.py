"""Save and load a GUI session as a .metbit project: a zip of JSON and Parquet, never pickle.

Fitted models are not stored (pickled objects would run code when a shared project is opened).
Data stages restore directly; analyses re-run from their saved params.
"""
import io
import json
import logging
import zipfile

import pandas as pd

from .annotations import parse_annotation_list
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
    manifest: dict = {
        "format": FORMAT,
        "version": VERSION,
        "stages": {},
        "style": session.style,
        "stage_styles": session.stage_styles,
        "annotations": session.annotations,
    }
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
    # before analyses re-run, so the Annotate result shares this list; re-sanitized like any import
    fresh.annotations = parse_annotation_list(manifest.get("annotations") or [])
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
    session.style = manifest.get("style") or {}
    session.annotations = fresh.annotations
    session.stage_styles = manifest.get("stage_styles") or {}
