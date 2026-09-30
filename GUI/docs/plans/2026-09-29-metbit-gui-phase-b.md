# metbit GUI Phase B (PLS-R, LDA, ICA, HCA, Group tests) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add metbit's extended multivariate methods (PLS-R, LDA, ICA, HCA) and multi-group statistics (ANOVA / Kruskal-Wallis with ranked features and boxplots) to the GUI.

**Architecture:** New `stages/multivariate.py` holds PLS-R, LDA, ICA and HCA; `stages/stats.py` gains a Group tests stage. The registry interleaves them in rail order. `resolve_schema` learns two things: flattening `Optional[X]` (`anyOf [X, null]`) into plain `X` so the form renders a normal input, and a `numeric_columns` option kind for the PLS-R target. Group tests rank features with the same scipy tests metbit uses (`f_oneway`, `kruskal`), vectorized across a binned matrix, then use metbit's `ANOVAStats` / `KruskalStats` for the chosen feature's boxplot. Spec: `GUI/docs/specs/2026-09-29-metbit-gui-design.md`.

**Tech Stack:** Same as Phase A. New calls: `metbit.analysis.multivariate.{plsr, lda, ica, hca}`, `metbit.stats.multitest.{ANOVAStats, KruskalStats, _apply_correction}`, `scipy.stats.{f_oneway, kruskal}` (scipy 1.18, `axis=0`).

## Global Constraints

- All GUI code lives under `GUI/`; Python GUI tests under `tests/gui/`.
- Never use the em dash character or the section sign anywhere.
- Commit messages carry no `Co-Authored-By` trailer.
- Keep every source file under 500 lines.
- Stage ids (exact, rail order after Phase B): `import`, `normalize`, `align`, `pca`, `opls`, `plsr`, `lda`, `ica`, `hca`, `stocsy`, `volcano`, `grouptests`.
- Rail sections: `Data` (import, normalize, align), `Analysis` (pca, opls, plsr, lda, ica, hca, stocsy), `Statistics` (volcano, grouptests).
- Every analysis and statistics stage: `requires="import"`, `after=list(DATA_ORDER)`.
- Python GUI tests: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`. Renderer: `cd GUI && npm run typecheck && npm test`. E2E: `cd GUI && npm run e2e`.

## File Map

```
GUI/server/metbit_server/stages/
  base.py            REPLACE: nullable flattening, numeric_columns option kind
  multivariate.py    NEW: PLS-R, LDA, ICA, HCA
  stats.py           REPLACE: + bin_matrix(), Group tests
  __init__.py        REPLACE: registry order
tests/gui/
  test_gui_schema.py        NEW
  test_gui_multivariate.py  NEW
  test_gui_group_tests.py   NEW
GUI/renderer/App.tsx        MODIFY: one tab label
GUI/e2e/smoke.spec.ts       REPLACE: + Phase B test
```

---

### Task 1: Schema: nullable fields and numeric column options

**Files:**
- Modify (replace): `GUI/server/metbit_server/stages/base.py`
- Test: `tests/gui/test_gui_schema.py`

**Interfaces:**
- Consumes: Phase A `base.py` (unchanged names: `Stage`, `Dataset`, `DATA_ORDER`, `working`, `groups`, `two_groups`, `spectra_figure`, `option_values`, `resolve_schema`).
- Produces:
  - `option_values(session, "numeric_columns") -> list[str] | None`: metadata columns whose every value parses as a number; `None` when there are none or before import.
  - `resolve_schema` replaces `anyOf: [X, {"type": "null"}]` with the keys of `X` (so `Optional[float]` renders as `type: "number"`).

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_schema.py`:

```python
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from metbit_server.session import Session
from metbit_server.stages import resolve_schema
from metbit_server.stages.base import Stage, option_values


class Tagged(BaseModel):
    value: Optional[float] = Field(None, title="Value")
    target: str = Field("", json_schema_extra={"options": "numeric_columns"})


TAGGED = Stage(id="tagged", title="Tagged", params=Tagged, run=lambda p, s, prog: None)


def test_optional_fields_flatten_to_their_plain_type():
    prop = resolve_schema(TAGGED, Session(["import"]))["properties"]["value"]
    assert prop["type"] == "number"
    assert "anyOf" not in prop
    assert prop["title"] == "Value"


def test_numeric_columns_option(run_all, csv_path, tmp_path):
    df = pd.read_csv(csv_path)
    df.insert(2, "Age", np.linspace(20, 70, 12))
    df.insert(3, "Site", ["x", "y"] * 6)
    path = tmp_path / "age.csv"
    df.to_csv(path, index=False)
    s = run_all(("import", {"path": str(path)}))
    assert option_values(s, "numeric_columns") == ["Age"]
    prop = resolve_schema(TAGGED, s)["properties"]["target"]
    assert (prop["enum"], prop["default"]) == (["Age"], "Age")


def test_numeric_columns_none_when_absent(run_all):
    assert option_values(run_all(), "numeric_columns") is None
    assert option_values(Session(["import"]), "numeric_columns") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_schema.py --no-cov -q -p no:warnings`
Expected: 3 failed (`anyOf` still present; `numeric_columns` returns `None`).

- [ ] **Step 3: Implement**

Replace `GUI/server/metbit_server/stages/base.py` with:

```python
"""Shared stage types and helpers for reading the working matrix."""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel

from metbit.viz.spectra import SpectraPlot

from ..session import Session

Progress = Callable[[float], None]

DATA_ORDER = ["import", "normalize", "align"]
MAX_PLOT_POINTS = 8000


@dataclass
class Stage:
    id: str
    title: str
    params: type[BaseModel]
    run: Callable[[Any, Session, Progress], Any]
    section: str = "Analysis"
    table: Optional[Callable[[Any], pd.DataFrame]] = None
    figures: dict[str, Callable[[Any], go.Figure]] = field(default_factory=dict)
    requires: Optional[str] = None
    after: list[str] = field(default_factory=list)
    # view name -> where a clicked point keeps its ppm: "x" or "customdata" (index 0)
    ppm_views: dict[str, str] = field(default_factory=dict)


@dataclass
class Dataset:
    X: pd.DataFrame                        # samples x features, float ppm column labels
    meta: pd.DataFrame                     # every non-ppm column
    labels: pd.Series                      # group name per sample (str)
    before: Optional[pd.DataFrame] = None  # input matrix, for before / after views


def working(session: Session, upto: Optional[str] = None) -> Dataset:
    """Newest `done` data stage, considering only stages earlier than `upto`."""
    order = DATA_ORDER if upto is None else DATA_ORDER[: DATA_ORDER.index(upto)]
    for sid in reversed(order):
        if session.status.get(sid) == "done":
            return session.outputs[sid]
    raise ValueError("Import data first")


def groups(session: Session) -> pd.Series:
    return session.output("import").labels


def two_groups(session: Session, a: str, b: str) -> tuple[pd.DataFrame, pd.Series]:
    """Working matrix and labels restricted to groups `a` and `b`."""
    if a == b:
        raise ValueError("Choose two different groups")
    g = groups(session)
    missing = [x for x in (a, b) if x not in set(g)]
    if missing:
        raise ValueError(f"Unknown group(s): {missing}. Groups: {sorted(set(g))}")
    mask = g.isin([a, b]).to_numpy()
    X = working(session).X
    return X[mask].reset_index(drop=True), g[mask].reset_index(drop=True)


def spectra_figure(X: pd.DataFrame, labels: pd.Series, title: str) -> go.Figure:
    # ponytail: display-only decimation keeps figure JSON small; full data stays in the session
    step = max(1, X.shape[1] // MAX_PLOT_POINTS)
    return SpectraPlot(X.iloc[:, ::step], label=list(labels)).overlay(title=title)


def _numeric_columns(meta: pd.DataFrame) -> list[str]:
    return [str(c) for c in meta.columns if pd.to_numeric(meta[c], errors="coerce").notna().all()]


def option_values(session: Session, kind: str) -> Optional[list[str]]:
    if session.status.get("import") != "done":
        return None
    meta = session.output("import").meta
    if kind == "groups":
        return sorted(set(groups(session)))
    if kind == "columns":
        return [str(c) for c in meta.columns]
    if kind == "numeric_columns":
        return _numeric_columns(meta) or None
    return None


def _flatten_nullable(prop: dict) -> None:
    """Optional[X] arrives as anyOf [X, null]; give the form the plain X schema."""
    options = prop.get("anyOf")
    if options and len(options) == 2 and {"type": "null"} in options:
        prop.update(next(o for o in options if o != {"type": "null"}))
        del prop["anyOf"]


def resolve_schema(stage: Stage, session: Session) -> dict:
    """Param JSON schema with data-driven `enum` and `default` for tagged fields."""
    schema = stage.params.model_json_schema()
    for prop in schema.get("properties", {}).values():
        _flatten_nullable(prop)
        values = option_values(session, prop.get("options", ""))
        if values:
            prop["enum"] = values
            if prop.get("default") not in values:
                prop["default"] = values[min(prop.get("pick", 0), len(values) - 1)]
    return schema
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `46 passed`.

- [ ] **Step 5: Commit**

```bash
git add GUI/server tests/gui
git commit -m "feat(gui): flatten optional fields and offer numeric metadata columns in schemas"
```

---

### Task 2: PLS-R, LDA, ICA and HCA stages

**Files:**
- Create: `GUI/server/metbit_server/stages/multivariate.py`
- Modify (replace): `GUI/server/metbit_server/stages/__init__.py`
- Modify: `tests/gui/test_gui_volcano.py` (Phase A full-registry assertion)
- Test: `tests/gui/test_gui_multivariate.py`

**Interfaces:**
- Consumes: `Stage`, `DATA_ORDER`, `working`, `groups` from `base.py`.
- Produces:
  - `PLSRParams(target: str = "" [options numeric_columns], n_components: int = 2 (1..10), scaling="pareto")`; stage `plsr` / `PLS-R`; views `predicted`, `scores`; table = one row with `R2`, `Q2`, `RMSE`. Raises `ValueError` "Target column '<t>' not found..." / "Target column '<t>' must be numeric with no missing values".
  - `LDAParams(n_components: int = 0 (0..10, 0 = groups - 1), scaling)`; `LDAResult(model, lds: list[str])`; stage `lda` / `LDA`; views `scores`, `loadings` (`ppm_views={"loadings": "x"}`); table = explained variance. Raises "LDA needs at least two groups" / "LDA allows at most <k-1> components for <k> groups". With two groups the scores view is a one-axis strip plot of LD1.
  - `ICAParams(n_components: int = 3 (2..10), scaling, max_iter: int = 1000 (100..10000))`; `ICAResult(model, labels)`; stage `ica` / `ICA`; views `components`, `mixing` (`ppm_views={"mixing": "x"}`); table = sample component scores.
  - `HCAParams(method="ward" | "average" | "complete" | "single", metric="euclidean" | "correlation" | "cityblock", n_clusters: int = 3 (2..20), scaling)`; validator: "Ward linkage requires the euclidean metric"; `HCAResult(model, n_clusters, table)`; stage `hca` / `HCA`; views `dendrogram`, `heatmap`; table = import metadata plus `Cluster`. Raises "n_clusters (<n>) exceeds the number of samples (<m>)".
  - `MULTIVARIATE_STAGES: list[Stage]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_multivariate.py`:

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from metbit_server.stages import STAGES, HCAParams, resolve_schema

BY_ID = {s.id: s for s in STAGES}


@pytest.fixture
def csv3(tmp_path, csv_path):
    """Three groups of four plus a numeric Age column."""
    df = pd.read_csv(csv_path)
    df["Group"] = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    df.insert(2, "Age", np.linspace(20, 70, 12))
    path = tmp_path / "three.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_analysis_rail_order():
    assert [s.id for s in STAGES if s.section == "Analysis"] == ["pca", "opls", "plsr", "lda", "ica", "hca", "stocsy"]
    for sid in ("plsr", "lda", "ica", "hca"):
        assert BY_ID[sid].requires == "import"
        assert BY_ID[sid].after == ["import", "normalize", "align"]


def test_plsr_target_defaults_to_numeric_column_and_runs(run_all, csv3):
    s = run_all(("import", {"path": csv3}))
    assert resolve_schema(BY_ID["plsr"], s)["properties"]["target"]["default"] == "Age"
    s = run_all(("import", {"path": csv3}), ("plsr", {"target": "Age"}))
    model = s.output("plsr")
    assert list(BY_ID["plsr"].table(model).columns) == ["R2", "Q2", "RMSE"]
    for view in ("predicted", "scores"):
        assert isinstance(BY_ID["plsr"].figures[view](model), go.Figure)


@pytest.mark.parametrize("target,msg", [("Sample", "must be numeric"), ("Weight", "not found")])
def test_plsr_rejects_bad_targets(run_all, csv3, target, msg):
    with pytest.raises(ValueError, match=msg):
        run_all(("import", {"path": csv3}), ("plsr", {"target": target}))


def test_lda_three_groups(run_all, csv3):
    r = run_all(("import", {"path": csv3}), ("lda", {})).output("lda")
    assert r.lds == ["LD1", "LD2"]
    for view in ("scores", "loadings"):
        assert isinstance(BY_ID["lda"].figures[view](r), go.Figure)
    assert BY_ID["lda"].ppm_views == {"loadings": "x"}


def test_lda_two_groups_uses_a_strip_plot(run_all):
    r = run_all(("lda", {})).output("lda")
    assert r.lds == ["LD1"]
    assert isinstance(BY_ID["lda"].figures["scores"](r), go.Figure)


def test_lda_rejects_too_many_components(run_all, csv3):
    with pytest.raises(ValueError, match="at most 2 components for 3 groups"):
        run_all(("import", {"path": csv3}), ("lda", {"n_components": 5}))


def test_ica(run_all):
    r = run_all(("ica", {"n_components": 3})).output("ica")
    assert BY_ID["ica"].table(r).shape == (12, 3)
    for view in ("components", "mixing"):
        assert isinstance(BY_ID["ica"].figures[view](r), go.Figure)


def test_hca_table_and_views(run_all, csv3):
    r = run_all(("import", {"path": csv3}), ("hca", {"n_clusters": 3})).output("hca")
    table = BY_ID["hca"].table(r)
    assert list(table.columns) == ["Sample", "Group", "Age", "Cluster"]
    assert table["Cluster"].nunique() <= 3
    for view in ("dendrogram", "heatmap"):
        assert isinstance(BY_ID["hca"].figures[view](r), go.Figure)


def test_hca_validation(run_all):
    with pytest.raises(ValidationError, match="Ward linkage requires the euclidean metric"):
        HCAParams(method="ward", metric="correlation")
    assert HCAParams(method="average", metric="correlation").metric == "correlation"
    with pytest.raises(ValueError, match="exceeds the number of samples"):
        run_all(("hca", {"n_clusters": 20}))
```

In `tests/gui/test_gui_volcano.py`, the Phase A test pins the whole 7-stage registry, which this task extends. Replace:

```python
def test_registry_order_and_sections():
    assert [s.id for s in STAGES] == ["import", "normalize", "align", "pca", "opls", "stocsy", "volcano"]
    assert [s.section for s in STAGES] == ["Data"] * 3 + ["Analysis"] * 3 + ["Statistics"]
```

with:

```python
def test_registry_order_and_sections():
    assert [s.id for s in STAGES][:3] == ["import", "normalize", "align"]
    assert all(s.section == "Data" for s in STAGES[:3])
    assert BY_ID["volcano"].section == "Statistics"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_multivariate.py --no-cov -q -p no:warnings`
Expected: FAIL with `ImportError: cannot import name 'HCAParams'`.

- [ ] **Step 3: Implement the stages**

Create `GUI/server/metbit_server/stages/multivariate.py`:

```python
"""Extended multivariate stages: PLS-R, LDA, ICA, HCA."""
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field, model_validator

from metbit.analysis.multivariate import hca, ica, lda, plsr

from ..session import Session
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


def _lda_scores(r: LDAResult) -> go.Figure:
    if len(r.lds) >= 2:
        return r.model.plot_lda_scores(ld=r.lds)
    # two groups give a single discriminant: metbit's 2D plot needs LD2
    return px.strip(r.model.get_scores(), x="Group", y="LD1", color="Group", title="LDA scores (LD1)")


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
            "components": lambda r: r.model.plot_components(ic=["IC1", "IC2"], color_=list(r.labels)),
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
            "heatmap": lambda r: r.model.plot_heatmap(n_clusters=r.n_clusters),
        },
    ),
]
```

Replace `GUI/server/metbit_server/stages/__init__.py` with:

```python
"""Stage registry, in rail order."""
from .analysis import ANALYSIS_STAGES, OPLSParams, OPLSResult, PCAParams, PCAResult, STOCSYParams
from .base import DATA_ORDER, Dataset, Progress, Stage, resolve_schema
from .data import DATA_STAGES, AlignParams, ImportParams, NormalizeParams, parse_windows
from .multivariate import MULTIVARIATE_STAGES, HCAParams, ICAParams, LDAParams, PLSRParams
from .stats import STATS_STAGES, VolcanoParams

_analysis = {s.id: s for s in ANALYSIS_STAGES}

STAGES: list[Stage] = [
    *DATA_STAGES,
    _analysis["pca"],
    _analysis["opls"],
    *MULTIVARIATE_STAGES,
    _analysis["stocsy"],
    *STATS_STAGES,
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `56 passed`.

- [ ] **Step 5: Commit**

```bash
git add GUI/server tests/gui
git commit -m "feat(gui): add PLS-R, LDA, ICA and HCA stages"
```

---

### Task 3: Group tests (ANOVA / Kruskal-Wallis)

**Files:**
- Modify (replace): `GUI/server/metbit_server/stages/stats.py`, `GUI/server/metbit_server/stages/__init__.py`
- Test: `tests/gui/test_gui_group_tests.py`

**Interfaces:**
- Consumes: `groups`, `working`, `two_groups`, `DATA_ORDER` from `base.py`; `_flatten_nullable` behavior via `resolve_schema`.
- Produces:
  - `bin_matrix(X: pd.DataFrame, width: float) -> pd.DataFrame`: averages columns into `width`-ppm bins labelled by bin center (rounded to 6 dp), columns in descending ppm; `width <= 0` returns `X` itself.
  - `GroupTestParams(test="anova" | "kruskal", correction="fdr_bh" | "bonferroni" | "none", bin_width: float = 0.01 (0..1), feature_ppm: float | None = None [pick_ppm])`; `""` for `feature_ppm` means `None`.
  - `GroupTestResult(table, data, test, feature: str)`; `table` columns `ppm, statistic, p_value, p_adj` sorted by `p_value` (NaN last); `data` = `Group` + binned features with `str(ppm)` column names; `feature` = column name used by the boxplot.
  - Stage `grouptests` / `Group tests`, section `Statistics`, view `boxplot`, table = ranked table. Raises "Group tests need at least two groups".

- [ ] **Step 1: Write the failing tests**

Create `tests/gui/test_gui_group_tests.py`:

```python
import contextlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit.stats.multitest import ANOVAStats
from metbit_server.app import create_app
from metbit_server.stages import STAGES, GroupTestParams, bin_matrix, resolve_schema

BY_ID = {s.id: s for s in STAGES}


def test_full_registry_order():
    assert [s.id for s in STAGES] == [
        "import", "normalize", "align",
        "pca", "opls", "plsr", "lda", "ica", "hca", "stocsy",
        "volcano", "grouptests",
    ]
    assert BY_ID["grouptests"].section == "Statistics"


def test_bin_matrix():
    X = pd.DataFrame(np.arange(20.0).reshape(2, 10), columns=np.linspace(1.95, 0.05, 10))
    assert bin_matrix(X, 0) is X
    b = bin_matrix(X, 1.0)
    assert list(b.columns) == [1.5, 0.5]
    assert b.iloc[0].tolist() == [np.mean(range(0, 5)), np.mean(range(5, 10))]


def test_anova_ranks_the_spiked_feature_first_and_matches_metbit(run_all):
    s = run_all(("grouptests", {"bin_width": 0}))
    r = s.output("grouptests")
    spiked = s.output("import").X.columns[5]
    assert np.isclose(r.table["ppm"].iloc[0], spiked)
    assert r.feature == str(spiked)
    with contextlib.redirect_stdout(io.StringIO()):
        ref = ANOVAStats(r.data, x_col="Group", y_col=r.feature)
        ref.fit()
    assert np.isclose(r.table["p_value"].iloc[0], ref.get_anova_table()["p_value"].iloc[0])
    assert list(BY_ID["grouptests"].table(r).columns) == ["ppm", "statistic", "p_value", "p_adj"]
    assert isinstance(BY_ID["grouptests"].figures["boxplot"](r), go.Figure)


def test_kruskal_survives_constant_features(run_all, csv_path, tmp_path):
    df = pd.read_csv(csv_path)
    df[df.columns[10]] = 1.0  # a flat region, like an excluded water window
    path = tmp_path / "flat.csv"
    df.to_csv(path, index=False)
    r = run_all(("import", {"path": str(path)}), ("grouptests", {"test": "kruskal", "bin_width": 0})).output("grouptests")
    assert r.table["p_value"].isna().sum() == 1
    assert r.table["p_value"].iloc[-1] != r.table["p_value"].iloc[-1]  # NaN sorted last
    assert isinstance(BY_ID["grouptests"].figures["boxplot"](r), go.Figure)


def test_feature_choice_and_no_correction(run_all):
    r = run_all(("grouptests", {"bin_width": 0, "feature_ppm": 2.0, "correction": "none"})).output("grouptests")
    assert np.isclose(float(r.feature), 2.0, atol=0.13)  # nearest of 40 points spaced 0.256 ppm
    assert np.allclose(r.table["p_adj"], r.table["p_value"], equal_nan=True)


def test_blank_feature_means_top_hit():
    assert GroupTestParams(feature_ppm="").feature_ppm is None


def test_needs_two_groups(run_all, csv_path, tmp_path):
    df = pd.read_csv(csv_path)
    df["Group"] = "A"
    path = tmp_path / "one.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="at least two groups"):
        run_all(("import", {"path": str(path)}), ("grouptests", {}))


def test_schema_and_http_run(run_all, csv_path):
    prop = resolve_schema(BY_ID["grouptests"], run_all())["properties"]["feature_ppm"]
    assert prop["type"] == "number" and prop["pick_ppm"] is True
    c = TestClient(create_app(TOKEN))
    for sid, params in (("import", {"path": csv_path}), ("grouptests", {"feature_ppm": "", "bin_width": 0.5})):
        job = c.post(f"/stages/{sid}/run", json=params, headers=AUTH).json()["job_id"]
        assert "event: done" in c.get(f"/jobs/{job}/events", headers=AUTH).text
    page = c.get("/stages/grouptests/table", headers=AUTH).json()
    assert page["columns"] == ["ppm", "statistic", "p_value", "p_adj"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/gui/test_gui_group_tests.py --no-cov -q -p no:warnings`
Expected: FAIL with `ImportError: cannot import name 'GroupTestParams'`.

- [ ] **Step 3: Implement the stage**

Replace `GUI/server/metbit_server/stages/stats.py` with:

```python
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


def _boxplot(r: GroupTestResult) -> go.Figure:
    cls = ANOVAStats if r.test == "anova" else KruskalStats
    stats = cls(r.data, x_col="Group", y_col=r.feature)
    stats.fit()
    return stats.plot(title=f"{float(r.feature):.4f} ppm")


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
```

Replace `GUI/server/metbit_server/stages/__init__.py` with:

```python
"""Stage registry, in rail order."""
from .analysis import ANALYSIS_STAGES, OPLSParams, OPLSResult, PCAParams, PCAResult, STOCSYParams
from .base import DATA_ORDER, Dataset, Progress, Stage, resolve_schema
from .data import DATA_STAGES, AlignParams, ImportParams, NormalizeParams, parse_windows
from .multivariate import MULTIVARIATE_STAGES, HCAParams, ICAParams, LDAParams, PLSRParams
from .stats import STATS_STAGES, GroupTestParams, GroupTestResult, VolcanoParams, bin_matrix

_analysis = {s.id: s for s in ANALYSIS_STAGES}

STAGES: list[Stage] = [
    *DATA_STAGES,
    _analysis["pca"],
    _analysis["opls"],
    *MULTIVARIATE_STAGES,
    _analysis["stocsy"],
    *STATS_STAGES,
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/gui --no-cov -q -p no:warnings`
Expected: `64 passed`.

- [ ] **Step 5: Commit**

```bash
git add GUI/server tests/gui
git commit -m "feat(gui): add Group tests with vectorized ANOVA and Kruskal-Wallis ranking"
```

---

### Task 4: Renderer label and end-to-end coverage

**Files:**
- Modify: `GUI/renderer/App.tsx` (the `LABELS` line)
- Modify (replace): `GUI/e2e/smoke.spec.ts`

**Interfaces:**
- Consumes: stage titles `PLS-R`, `LDA`, `ICA`, `HCA`, `Group tests`; control ids `#f-target`; tab names `Predicted vs actual`, `Dendrogram`, `Heatmap`, `Boxplot`, `Table`.
- Produces: `npm run e2e` runs 3 tests.

- [ ] **Step 1: Tab label**

In `GUI/renderer/App.tsx`, replace:

```tsx
const LABELS: Record<string, string> = { vip: "VIP", "s-plot": "S-plot" };
```

with:

```tsx
const LABELS: Record<string, string> = { vip: "VIP", "s-plot": "S-plot", predicted: "Predicted vs actual" };
```

- [ ] **Step 2: Replace the e2e spec**

Replace `GUI/e2e/smoke.spec.ts` with:

```ts
import { _electron as electron, expect, test, type Page } from "@playwright/test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function makeCsv(groups = ["A", "B"], perGroup = 6): string {
  const ppm = Array.from({ length: 40 }, (_, i) => (10 - i * 0.25).toFixed(4));
  const n = groups.length * perGroup;
  const rows = Array.from({ length: n }, (_, i) => {
    const g = Math.floor(i / perGroup);
    const values = ppm.map((_, j) => (Math.abs(Math.sin(i * 7 + j)) + 1 + (j === 5 ? g * 1.5 : 0)).toFixed(5));
    return [`s${i}`, groups[g], (20 + i * 3).toFixed(1), ...values].join(",");
  });
  const file = join(mkdtempSync(join(tmpdir(), "metbit-e2e-")), "data.csv");
  writeFileSync(file, [["Sample", "Group", "Age", ...ppm].join(","), ...rows].join("\n"));
  return file;
}

const run = (win: Page) => win.getByRole("button", { name: "Run", exact: true }).click();
const stageButton = (win: Page, name: string) => win.locator(".rail button", { hasText: name });
const plot = (win: Page) => expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

async function launchWithImport(csv: string) {
  const app = await electron.launch({ args: ["."] });
  const win = await app.firstWindow();
  await win.fill("#f-path", csv);
  await run(win);
  await expect(win.locator("table.data")).toContainText("s11");
  return { app, win };
}

test("import a CSV and run PCA", async () => {
  const { app, win } = await launchWithImport(makeCsv());
  await stageButton(win, "PCA").click();
  await run(win);
  await plot(win);
  await app.close();
});

test("normalize, OPLS-DA, STOCSY and volcano", async () => {
  const { app, win } = await launchWithImport(makeCsv());

  await stageButton(win, "Normalize").click();
  await run(win);
  await plot(win);

  await stageButton(win, "OPLS-DA").click();
  await expect(win.locator("#f-group_a")).toHaveValue("A");
  await expect(win.locator("#f-group_b")).toHaveValue("B");
  await run(win);
  await plot(win);
  await win.getByRole("tab", { name: "VIP" }).click();
  await plot(win);

  await stageButton(win, "STOCSY").click();
  await win.fill("#f-anchor_ppm", "5");
  await run(win);
  await plot(win);

  await stageButton(win, "Volcano").click();
  await run(win);
  await expect(win.getByRole("tab", { name: "Volcano" })).toBeVisible();
  await plot(win);

  await app.close();
});

test("PLS-R, LDA, ICA, HCA and group tests", async () => {
  const { app, win } = await launchWithImport(makeCsv(["A", "B", "C"], 4));

  await stageButton(win, "PLS-R").click();
  await expect(win.locator("#f-target")).toHaveValue("Age");
  await run(win);
  await expect(win.getByRole("tab", { name: "Predicted vs actual" })).toBeVisible();
  await plot(win);

  for (const name of ["LDA", "ICA"]) {
    await stageButton(win, name).click();
    await run(win);
    await plot(win);
  }

  await stageButton(win, "HCA").click();
  await run(win);
  await plot(win);
  await win.getByRole("tab", { name: "Heatmap" }).click();
  await plot(win);

  await stageButton(win, "Group tests").click();
  await run(win);
  await plot(win);
  await win.getByRole("tab", { name: "Table" }).click();
  await expect(win.locator("table.data")).toContainText("p_adj");

  await app.close();
});
```

- [ ] **Step 3: Run everything**

Run from the repo root:
```bash
python3 -m pytest -q -p no:cacheprovider && (cd GUI && npm run typecheck && npm test && npm run e2e)
```
Expected: repo suite passes with coverage at or above 99%; typecheck exits 0; `13 passed`; `3 passed`.

- [ ] **Step 4: Commit**

```bash
git add GUI/renderer/App.tsx GUI/e2e/smoke.spec.ts
git commit -m "test(gui): cover PLS-R, LDA, ICA, HCA and group tests end to end"
```
