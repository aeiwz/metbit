"""Analysis stages: PCA, OPLS-DA, STOCSY."""
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import plotly.graph_objects as go
from pydantic import BaseModel, Field, model_validator

from metbit import opls_da, pca
from metbit.analysis.stocsy import STOCSY

from ..session import Session
from .style import PlotStyle, colors, symbols
from .base import DATA_ORDER, Dataset, Progress, Stage, groups, not_run_figure, spectra_figure, two_groups, working

Scaling = Literal["pareto", "uv", "mean"]


def scores_style(style: Optional[PlotStyle], labels: Any) -> dict:
    """metbit's own scores-plot styling arguments (PCA and OPLS-DA share them)."""
    if style is None or labels is None:
        return {}
    return {
        "color_dict": colors(style),
        "symbol_": labels,
        "symbol_dict": symbols(style),
        "marker_size": style.marker_size,
        "marker_opacity": style.marker_opacity,
        "font_size": style.font_size,
        "individual_ellipse": style.show_ellipses,
    }


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
    labels: Any = None  # group per sample, for styling


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
    return PCAResult(model=model, pcs=[f"PC{p.pc_x}", f"PC{p.pc_y}"], labels=groups(session))


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
    labels: Any = None  # group per sample in the two-group subset


def run_opls(p: OPLSParams, session: Session, progress: Progress) -> OPLSResult:
    X, y = two_groups(session, p.group_a, p.group_b)
    model = opls_da(X=X, y=y, features_name=list(X.columns), scaling_method=p.scaling, kfold=p.kfold)
    model.fit()
    model.vip_scores()
    if p.permutations:
        progress(0.3)
        model.permutation_test(n_permutations=p.permutations, cv=p.kfold, n_jobs=-1, verbose=0)
    progress(1.0)
    return OPLSResult(model=model, vip_threshold=p.vip_threshold, permuted=p.permutations > 0, labels=y)


def _permutation_view(r: OPLSResult) -> go.Figure:
    if not r.permuted:
        return not_run_figure("Permutation test not run. Set Permutations above 0 and run again.")
    return r.model.plot_hist()


# STOCSY ---------------------------------------------------------------------

class STOCSYParams(BaseModel):
    anchor_ppm: float = Field(title="Anchor (ppm)", json_schema_extra={"pick_ppm": True})
    p_threshold: float = Field(1e-4, gt=0, lt=1, title="p threshold")


def run_stocsy(p: STOCSYParams, session: Session, progress: Progress) -> go.Figure:
    data = working(session)
    X = data.X
    lo, hi = float(X.columns.min()), float(X.columns.max())
    if not lo <= p.anchor_ppm <= hi:
        raise ValueError(f"Anchor {p.anchor_ppm} ppm is outside the spectrum ({lo:.2f} to {hi:.2f} ppm)")
    cache = data.cache.setdefault("stocsy", {})  # like the Dash app's lru_cache, scoped to this matrix
    key = (p.anchor_ppm, p.p_threshold)
    if key not in cache:
        cache[key] = STOCSY(X, p.anchor_ppm, p.p_threshold)
    progress(1.0)
    return cache[key]


def sample_indices(n: int) -> list[int]:
    """Spectra to draw, as metbit's STOCSY app: all up to 5, 20% from 100, else up to 30; seeded."""
    if n <= 5:
        return list(range(n))
    k = int(0.2 * n) if n >= 100 else min(30, n)
    return sorted(np.random.default_rng(0).choice(n, size=k, replace=False).tolist())


def stocsy_spectra(d: Dataset, style=None) -> go.Figure:
    idx = sample_indices(len(d.X))
    return spectra_figure(d.X.iloc[idx], d.labels.iloc[idx], "Click a peak to set the anchor", style)


ANALYSIS_STAGES: list[Stage] = [
    Stage(
        id="pca",
        title="PCA",
        params=PCAParams,
        run=run_pca,
        table=lambda r: r.model.get_explained_variance(),
        figures={
            "scores": lambda r, st=None: r.model.plot_pca_scores(pc=r.pcs, **scores_style(st, r.labels)),
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
            "scores": lambda r, st=None: r.model.plot_oplsda_scores(**scores_style(st, r.labels)),
            "s-plot": lambda r: r.model.plot_s_scores(),
            "vip": lambda r, st=None: r.model.vip_plot(
                threshold=r.vip_threshold, **({"marker_size": st.marker_size, "font_size": st.font_size} if st else {})
            ),
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
        # a copy: rendering restyles and labels the figure, and the stored one is also the cache entry
        figures={"correlation": lambda fig: go.Figure(fig)},
        pre_views={"spectra": stocsy_spectra},
        requires="import",
        after=list(DATA_ORDER),
        ppm_views={"spectra": "x", "correlation": "x"},
    ),
]
