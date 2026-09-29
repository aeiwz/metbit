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
