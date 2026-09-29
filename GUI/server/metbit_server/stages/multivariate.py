"""Extended multivariate stages: PLS-R, LDA, ICA, HCA."""
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field, model_validator

from metbit.analysis.multivariate import hca, ica, lda, plsr

from ..session import Session
from .style import colors
from .base import DATA_ORDER, Progress, Stage, groups, working

Scaling = Literal["pareto", "uv", "mean"]


# PLS-R ----------------------------------------------------------------------

class PLSRParams(BaseModel):
    target: str = Field(
        "",
        title="Target column",
        description="A numeric metadata column, such as age or dose.",
        json_schema_extra={"options": "numeric_columns"},
    )
    n_components: int = Field(2, ge=1, le=10, title="Components")
    scaling: Scaling = Field("pareto", title="Scaling")


def run_plsr(p: PLSRParams, session: Session, progress: Progress) -> Any:
    meta = session.output("import").meta
    if p.target not in meta.columns:
        raise ValueError(f"Target column '{p.target}' not found. Columns: {list(meta.columns)}")
    y = pd.to_numeric(meta[p.target], errors="coerce")
    if y.isna().any():
        raise ValueError(f"Target column '{p.target}' must be numeric with no missing values")
    X = working(session).X
    model = plsr(
        X=X,
        y=y.reset_index(drop=True),
        features_name=list(X.columns),
        n_components=p.n_components,
        scaling_method=p.scaling,
    )
    model.fit()
    progress(1.0)
    return model


# LDA ------------------------------------------------------------------------

class LDAParams(BaseModel):
    n_components: int = Field(
        0, ge=0, le=10, title="Components", description="0 uses the maximum: number of groups minus 1."
    )
    scaling: Scaling = Field("pareto", title="Scaling")


@dataclass
class LDAResult:
    model: Any
    lds: list[str]


def run_lda(p: LDAParams, session: Session, progress: Progress) -> LDAResult:
    X, y = working(session).X, groups(session)
    k = y.nunique()
    if k < 2:
        raise ValueError("LDA needs at least two groups")
    if p.n_components > k - 1:
        raise ValueError(f"LDA allows at most {k - 1} components for {k} groups")
    model = lda(
        X=X,
        y=y,
        features_name=list(X.columns),
        n_components=p.n_components or None,
        scaling_method=p.scaling,
    )
    model.fit()
    lds = [c for c in model.get_scores().columns if str(c).startswith("LD")][:2]
    progress(1.0)
    return LDAResult(model=model, lds=lds)


def _lda_scores(r: LDAResult, style=None) -> go.Figure:
    palette = colors(style) if style else None
    if len(r.lds) >= 2:
        extra = {"color_dict": palette, "marker_size": style.marker_size, "font_size": style.font_size} if style else {}
        return r.model.plot_lda_scores(ld=r.lds, **extra)
    # two groups give a single discriminant: metbit's 2D plot needs LD2
    return px.strip(
        r.model.get_scores(), x="Group", y="LD1", color="Group", title="LDA scores (LD1)", color_discrete_map=palette or {}
    )


# ICA ------------------------------------------------------------------------

class ICAParams(BaseModel):
    n_components: int = Field(3, ge=2, le=10, title="Components")
    scaling: Scaling = Field("pareto", title="Scaling")
    max_iter: int = Field(1000, ge=100, le=10000, title="Max iterations")


@dataclass
class ICAResult:
    model: Any
    labels: pd.Series


def run_ica(p: ICAParams, session: Session, progress: Progress) -> ICAResult:
    model = ica(
        X=working(session).X,
        n_components=p.n_components,
        max_iter=p.max_iter,
        scaling_method=p.scaling,
    )
    model.fit()
    progress(1.0)
    return ICAResult(model=model, labels=groups(session))


# HCA ------------------------------------------------------------------------

class HCAParams(BaseModel):
    method: Literal["ward", "average", "complete", "single"] = Field("ward", title="Linkage")
    metric: Literal["euclidean", "correlation", "cityblock"] = Field("euclidean", title="Distance")
    n_clusters: int = Field(3, ge=2, le=20, title="Clusters")
    scaling: Scaling = Field("pareto", title="Scaling")

    @model_validator(mode="after")
    def _ward_needs_euclidean(self) -> "HCAParams":
        if self.method == "ward" and self.metric != "euclidean":
            raise ValueError("Ward linkage requires the euclidean metric")
        return self


@dataclass
class HCAResult:
    model: Any
    n_clusters: int
    table: pd.DataFrame


def run_hca(p: HCAParams, session: Session, progress: Progress) -> HCAResult:
    X = working(session).X
    if p.n_clusters > len(X):
        raise ValueError(f"n_clusters ({p.n_clusters}) exceeds the number of samples ({len(X)})")
    model = hca(
        X=X,
        label=list(groups(session)),
        features_name=list(X.columns),
        method=p.method,
        metric=p.metric,
        scaling_method=p.scaling,
    )
    model.fit()
    clusters = model.get_cluster_labels(n_clusters=p.n_clusters).to_numpy()
    table = session.output("import").meta.reset_index(drop=True).assign(Cluster=clusters)
    progress(1.0)
    return HCAResult(model=model, n_clusters=p.n_clusters, table=table)


def _hca_heatmap(r: HCAResult) -> go.Figure:
    fig = r.model.plot_heatmap(n_clusters=r.n_clusters)
    fig.update_yaxes(anchor="x2")  # sample labels left of the cluster strip instead of drawn over it
    return fig


def _analysis(**kw) -> Stage:
    return Stage(requires="import", after=list(DATA_ORDER), **kw)


MULTIVARIATE_STAGES: list[Stage] = [
    _analysis(
        id="plsr",
        title="PLS-R",
        params=PLSRParams,
        run=run_plsr,
        table=lambda m: pd.DataFrame([m.get_metrics()]),
        figures={
            "predicted": lambda m: m.plot_predicted_vs_actual(),
            "scores": lambda m: m.plot_scores(),
        },
    ),
    _analysis(
        id="lda",
        title="LDA",
        params=LDAParams,
        run=run_lda,
        table=lambda r: r.model.get_explained_variance(),
        figures={"scores": _lda_scores, "loadings": lambda r: r.model.plot_loading_(ld=r.lds)},
        ppm_views={"loadings": "x"},
    ),
    _analysis(
        id="ica",
        title="ICA",
        params=ICAParams,
        run=run_ica,
        table=lambda r: r.model.get_components(),
        figures={
            "components": lambda r, st=None: r.model.plot_components(
                ic=["IC1", "IC2"], color_=list(r.labels), **({"color_dict": colors(st), "font_size": st.font_size} if st else {})
            ),
            "mixing": lambda r: r.model.plot_mixing_(ic=["IC1", "IC2"]),
        },
        ppm_views={"mixing": "x"},
    ),
    _analysis(
        id="hca",
        title="HCA",
        params=HCAParams,
        run=run_hca,
        table=lambda r: r.table,
        figures={
            "dendrogram": lambda r: r.model.plot_dendrogram(),
            "heatmap": _hca_heatmap,
        },
    ),
]
