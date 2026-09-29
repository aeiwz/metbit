"""Statistics stages: Volcano."""
from typing import Literal

from pydantic import BaseModel, Field

from metbit.stats.multitest import VolcanoPlot

from ..session import Session
from .base import DATA_ORDER, Progress, Stage, two_groups


def _group_field(title: str, pick: int):
    return Field("", title=title, json_schema_extra={"options": "groups", "pick": pick})


class VolcanoParams(BaseModel):
    group_a: str = _group_field("Group A (reference)", 0)
    group_b: str = _group_field("Group B", 1)
    fc_threshold: float = Field(1.0, ge=0, title="log2 fold-change threshold")
    p_threshold: float = Field(0.05, gt=0, lt=1, title="p threshold")
    correction: Literal["fdr_bh", "bonferroni", "none"] = Field("fdr_bh", title="Multiple testing")


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
]
