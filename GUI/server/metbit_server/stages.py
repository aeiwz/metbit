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
