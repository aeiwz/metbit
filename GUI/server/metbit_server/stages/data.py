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
