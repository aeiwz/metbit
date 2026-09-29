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
    permutations: int = Field(
        0,
        ge=0,
        le=1000,
        title="Permutations",
        # mirrors metbit's own UserWarning so GUI users see it too
        description="0 skips it. Exploratory only: metbit permutes the underlying PLS score, not the OPLS Q2.",
    )


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
