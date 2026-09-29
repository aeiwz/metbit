"""Statistics stages: Volcano, Group tests (ANOVA / Kruskal-Wallis)."""
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field, field_validator
from scipy.stats import f_oneway, kruskal

from metbit.stats.multitest import ANOVAStats, KruskalStats, VolcanoPlot, _apply_correction

from ..session import Session
from .style import colors
from .base import DATA_ORDER, Progress, Stage, groups, two_groups, working

Correction = Literal["fdr_bh", "bonferroni", "none"]


def _group_field(title: str, pick: int):
    return Field("", title=title, json_schema_extra={"options": "groups", "pick": pick})


# Volcano --------------------------------------------------------------------

class VolcanoParams(BaseModel):
    group_a: str = _group_field("Group A (reference)", 0)
    group_b: str = _group_field("Group B", 1)
    fc_threshold: float = Field(1.0, ge=0, title="log2 fold-change threshold")
    p_threshold: float = Field(0.05, gt=0, lt=1, title="p threshold")
    correction: Correction = Field("fdr_bh", title="Multiple testing")


def run_volcano(p: VolcanoParams, session: Session, progress: Progress) -> VolcanoPlot:
    X, y = two_groups(session, p.group_a, p.group_b)
    df = X.copy()
    df.columns = [str(c) for c in X.columns]  # full precision: 4 dp can collide at 0.0003 ppm spacing
    value_cols = list(df.columns)
    df.insert(0, "Group", y.to_numpy())
    result = VolcanoPlot(
        df,
        group_col="Group",
        value_cols=value_cols,
        group_a=p.group_a,
        group_b=p.group_b,
        p_value_threshold=p.p_threshold,
        fc_threshold=p.fc_threshold,
        correct_p=None if p.correction == "none" else p.correction,
    )
    progress(1.0)
    return result


# Group tests ----------------------------------------------------------------

def bin_matrix(X: pd.DataFrame, width: float) -> pd.DataFrame:
    """Average columns into `width`-ppm bins labelled by bin center, high ppm first."""
    if width <= 0:
        return X
    centers = (np.floor(X.columns.to_numpy(dtype=float) / width) + 0.5) * width
    binned = X.T.groupby(np.round(centers, 6)).mean().T
    return binned[sorted(binned.columns, reverse=True)]


class GroupTestParams(BaseModel):
    test: Literal["anova", "kruskal"] = Field("anova", title="Test")
    correction: Correction = Field("fdr_bh", title="Multiple testing")
    bin_width: float = Field(
        0.01, ge=0, le=1, title="Bin width (ppm)", description="Features are averaged into bins first. 0 tests every point."
    )
    feature_ppm: Optional[float] = Field(
        None,
        title="Boxplot feature (ppm)",
        description="Empty shows the top-ranked feature.",
        json_schema_extra={"pick_ppm": True},
    )

    @field_validator("feature_ppm", mode="before")
    @classmethod
    def _blank_is_none(cls, v):
        return None if v == "" else v


@dataclass
class GroupTestResult:
    table: pd.DataFrame  # ranked: ppm, statistic, p_value, p_adj
    data: pd.DataFrame   # Group + binned features (str ppm names), for the boxplot
    test: str
    feature: str         # column of `data` shown in the boxplot


def run_group_tests(p: GroupTestParams, session: Session, progress: Progress) -> GroupTestResult:
    labels = groups(session)
    names = sorted(set(labels))
    if len(names) < 2:
        raise ValueError("Group tests need at least two groups")
    Xb = bin_matrix(working(session).X, p.bin_width)
    # same scipy tests that metbit's ANOVAStats / KruskalStats call, vectorized over features
    varying = (Xb.nunique() > 1).to_numpy()
    stat = np.full(Xb.shape[1], np.nan)
    pvals = np.full(Xb.shape[1], np.nan)
    if varying.any():
        arrays = [Xb.loc[(labels == n).to_numpy(), varying].to_numpy() for n in names]
        test = f_oneway if p.test == "anova" else kruskal
        s, pv = test(*arrays, axis=0)
        stat[varying], pvals[varying] = s, pv
    progress(0.6)
    ppm = Xb.columns.to_numpy(dtype=float)
    table = pd.DataFrame({
        "ppm": ppm,
        "statistic": stat,
        "p_value": pvals,
        "p_adj": _apply_correction(pvals, None if p.correction == "none" else p.correction),
    }).sort_values("p_value", na_position="last").reset_index(drop=True)
    target = table["ppm"].iloc[0] if p.feature_ppm is None else ppm[np.abs(ppm - p.feature_ppm).argmin()]
    data = Xb.copy()
    data.columns = [str(c) for c in ppm]
    data.insert(0, "Group", labels.to_numpy())
    progress(1.0)
    return GroupTestResult(table=table, data=data, test=p.test, feature=str(target))


def _boxplot(r: GroupTestResult, style=None) -> go.Figure:
    cls = ANOVAStats if r.test == "anova" else KruskalStats
    stats = cls(r.data, x_col="Group", y_col=r.feature)
    stats.fit()
    extra = {"custom_colors": colors(style), "font_size": style.font_size} if style else {}
    return stats.plot(title=f"{float(r.feature):.4f} ppm", **extra)


STATS_STAGES: list[Stage] = [
    Stage(
        id="volcano",
        title="Volcano",
        section="Statistics",
        params=VolcanoParams,
        run=run_volcano,
        table=lambda v: v.get_table(),
        figures={"volcano": lambda v: v.plot()},
        requires="import",
        after=list(DATA_ORDER),
    ),
    Stage(
        id="grouptests",
        title="Group tests",
        section="Statistics",
        params=GroupTestParams,
        run=run_group_tests,
        table=lambda r: r.table,
        figures={"boxplot": _boxplot},
        requires="import",
        after=list(DATA_ORDER),
    ),
]
