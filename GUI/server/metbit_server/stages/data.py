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
