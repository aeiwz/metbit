"""ASCA and per-feature repeated-measures tests (linear mixed model, Friedman)."""
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd
import patsy
import plotly.graph_objects as go
from pydantic import Field, field_validator
from scipy.stats import chi2, friedmanchisquare
from statsmodels.regression.mixed_linear_model import MixedLM

from metbit.stats.multitest import _apply_correction

from ..session import Session
from .analysis import Scaling
from .base import DATA_ORDER, Progress, Stage
from .stats import Correction, bin_matrix
from .timeseries import SECTION, Design, TimeParams, design, need_subject, profile_figure

INTERACTION = "Time × Group"


def _bin_field(default: float):
    return Field(default, ge=0, le=1, title="Bin width (ppm)", description="Features are averaged into bins first. 0 uses every point.")


def scale(X: np.ndarray, how: str) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    if how == "mean":
        return Xc
    sd = X.std(axis=0, ddof=1)
    sd[~(sd > 0)] = 1.0
    return Xc / (np.sqrt(sd) if how == "pareto" else sd)


def level_means(X: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Each row replaced by the mean of its level: the effect matrix of one factor."""
    D = np.eye(codes.max() + 1)[codes]
    counts = D.sum(axis=0)[:, None]
    return D @ ((D.T @ X) / np.where(counts > 0, counts, 1))


# ASCA -----------------------------------------------------------------------

class AscaParams(TimeParams):
    scaling: Scaling = Field("pareto", title="Scaling")
    permutations: int = Field(200, ge=0, le=5000, title="Permutations", description="Per effect. 0 skips the p-values.")
    bin_width: float = _bin_field(0.01)


@dataclass
class AscaResult:
    table: pd.DataFrame
    effects: dict                     # name -> {"scores": n x 2 (effect + residual), "loadings": 2 x p, "explained": [.,.]}
    combined: np.ndarray              # PC1 of time + interaction (with residual), per sample
    ppm: np.ndarray
    des: Design
    extra: dict = field(default_factory=dict)


def _pca(M: np.ndarray, E: np.ndarray) -> dict:
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    V = np.zeros((2, M.shape[1]))
    V[: min(2, len(Vt))] = Vt[:2]
    var = s**2 / (s**2).sum() if (s**2).sum() > 0 else np.zeros_like(s)
    explained = [float(v) * 100 for v in np.pad(var[:2], (0, max(0, 2 - len(var[:2]))))]
    return {"scores": (M + E) @ V.T, "loadings": V, "explained": explained}


def _within_subject(des: Design, rng: np.random.Generator):
    n = len(des.time)
    if des.subject is None:
        return lambda: rng.permutation(n)
    subjects = des.subject.to_numpy()
    blocks = [np.flatnonzero(subjects == s) for s in np.unique(subjects)]

    def shuffle() -> np.ndarray:
        rows = np.arange(n)
        for b in blocks:
            rows[b] = rng.permutation(b)
        return rows

    # ponytail: Group still permutes samples freely; permuting whole subjects is stricter when groups are small
    return shuffle


def run_asca(p: AscaParams, session: Session, progress: Progress) -> AscaResult:
    des, d = design(session, p)
    Xb = bin_matrix(d.X, p.bin_width)
    X = scale(Xb.to_numpy(dtype=float), p.scaling)
    t = des.position()
    names = sorted(set(des.labels))
    g = des.labels.map({n: i for i, n in enumerate(names)}).to_numpy()
    cell = t * len(names) + g

    def time_effect(rows):
        return level_means(X, t[rows])

    def group_effect(rows):
        return level_means(X, g[rows])

    main = time_effect(slice(None)) + (group_effect(slice(None)) if len(names) > 1 else 0)
    R = X - main  # interaction is tested on what the main effects leave

    def interaction_effect(rows):
        return level_means(R, cell[rows]) - level_means(R, t[rows]) - level_means(R, g[rows])

    specs = {"Time": time_effect}
    if len(names) > 1:
        # ponytail: level-mean ASCA; exact for balanced designs, approximate otherwise (ASCA+ uses a GLM)
        specs.update({"Group": group_effect, INTERACTION: interaction_effect})
    total = (X**2).sum()
    rng = np.random.default_rng(0)
    within = _within_subject(des, rng)
    matrices, rows = {}, []
    for i, (name, effect) in enumerate(specs.items()):
        M = effect(slice(None))
        matrices[name] = M
        ssq = (M**2).sum()
        pv = np.nan
        if p.permutations:
            # repeated measures: time and interaction shuffle only within each subject
            shuffle = within if name != "Group" else (lambda: rng.permutation(len(X)))
            hits = sum((effect(shuffle()) ** 2).sum() >= ssq for _ in range(p.permutations))
            pv = (hits + 1) / (p.permutations + 1)
        rows.append({"effect": name, "ssq_percent": ssq / total * 100, "p_value": pv})
        progress((i + 1) / (len(specs) + 1))
    E = X - sum(matrices.values())
    rows.append({"effect": "Residual", "ssq_percent": (E**2).sum() / total * 100, "p_value": np.nan})
    effects = {name: _pca(M, E) for name, M in matrices.items()}
    table = pd.DataFrame(rows)
    table["pc1_percent"] = [effects[n]["explained"][0] if n in effects else np.nan for n in table["effect"]]
    table["pc2_percent"] = [effects[n]["explained"][1] if n in effects else np.nan for n in table["effect"]]
    combined = _pca(matrices["Time"] + matrices.get(INTERACTION, 0), E)["scores"][:, 0]
    progress(1.0)
    return AscaResult(table, effects, combined, Xb.columns.to_numpy(dtype=float), des)


def _fmt_p(v: float) -> str:
    return "p n/a" if v != v else f"p = {v:.3g}"


def _effects(r: AscaResult, style=None) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=r.table["effect"], y=r.table["ssq_percent"], marker_color="#0072B2",
        text=[_fmt_p(v) if e != "Residual" else "" for e, v in zip(r.table["effect"], r.table["p_value"])],
        textposition="outside", cliponaxis=False,
    ))
    return fig.update_layout(title="ASCA effects", template="plotly_white", yaxis_title="Variation explained (%)")


def _scores(r: AscaResult, style=None) -> go.Figure:
    """PC1 of the time + interaction effects (ASCA+ combined), per group over time."""
    title = "Time + interaction"
    if INTERACTION in r.effects:
        title = f"{title} · {_fmt_p(r.table.set_index('effect').loc[INTERACTION, 'p_value'])}"
    return profile_figure(r.combined, r.des, "mean", "sem", title, style, ytitle="PC1 score")


def _loadings(r: AscaResult, style=None) -> go.Figure:
    fig = go.Figure()
    for name, e in r.effects.items():
        fig.add_trace(go.Scatter(x=r.ppm, y=e["loadings"][0], name=name, mode="lines"))
    return fig.update_layout(
        title="PC1 loadings per effect", template="plotly_white",
        xaxis={"autorange": "reversed", "title": "δ¹H (ppm)"}, yaxis_title="Loading",
    )


# Mixed model / Friedman -----------------------------------------------------

TERMS = {"time": "time", "group": "group", "time:group": "interaction"}


class MixedParams(TimeParams):
    test: Literal["mixed", "friedman"] = Field(
        "mixed",
        title="Test",
        description="Mixed: y ~ time × group + (1 | subject). Friedman: rank test across time points within subjects.",
    )
    time_as: Literal["categorical", "numeric"] = Field(
        "categorical",
        title="Time in the model",
        description="Numeric fits a linear trend (time values when they are numbers, else the time order).",
    )
    correction: Correction = Field("fdr_bh", title="Multiple testing")
    bin_width: float = _bin_field(0.02)
    feature_ppm: Optional[float] = Field(
        None,
        title="Profile feature (ppm)",
        description="Click the p-value plot below. Empty shows the top-ranked feature.",
        json_schema_extra={"pick_ppm": True},
    )

    @field_validator("feature_ppm", mode="before")
    @classmethod
    def _blank_is_none(cls, v):
        return None if v == "" else v


@dataclass
class MixedResult:
    table: pd.DataFrame   # ppm, then p / p_adj per term, ranked
    data: pd.DataFrame    # binned matrix (float ppm columns)
    des: Design
    feature: float
    test: str


def _mixed_pvalues(Y: np.ndarray, des: Design, numeric: bool, progress: Progress) -> dict[str, np.ndarray]:
    time = des.time
    if numeric:
        values = pd.to_numeric(time, errors="coerce")
        time = values if values.notna().all() else pd.Series(des.position(), dtype=float)
    else:
        time = pd.Categorical(time, categories=des.order)
    frame = pd.DataFrame({"time": time, "group": pd.Categorical(des.labels)})
    formula = "time * group" if frame["group"].nunique() > 1 else "time"
    dm = patsy.dmatrix(formula, frame, return_type="dataframe")
    slices = {TERMS[k]: v for k, v in dm.design_info.term_name_slices.items() if k in TERMS}
    out = {term: np.full(Y.shape[1], np.nan) for term in slices}
    exog, subjects = dm.to_numpy(), des.subject.to_numpy()
    failure: Optional[Exception] = None
    for j in range(Y.shape[1]):
        y = Y[:, j]
        if np.ptp(y) > 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # statsmodels' lbfgs fallback passes `disp`, which SciPy >= 1.18 rejects
                    fit = MixedLM(y, exog, groups=subjects).fit(reml=True, method=["bfgs", "cg", "nm"])
                k = exog.shape[1]
                b, V = np.asarray(fit.fe_params), np.asarray(fit.cov_params())[:k, :k]
                for term, sl in slices.items():
                    stat = float(b[sl] @ np.linalg.pinv(V[sl, sl]) @ b[sl])  # Wald test of the whole term
                    out[term][j] = chi2.sf(stat, sl.stop - sl.start)
            except Exception as e:  # one feature that cannot be fitted stays NaN
                failure = failure or e
        if j % 25 == 0:
            progress(0.05 + 0.9 * j / Y.shape[1])
    if failure is not None and all(np.isnan(v).all() for v in out.values()):
        raise ValueError(f"The mixed model could not be fitted: {failure}")
    return out


def _friedman_pvalues(Xb: pd.DataFrame, des: Design) -> dict[str, np.ndarray]:
    if len(des.order) < 3:
        raise ValueError("The Friedman test needs at least three time points")
    means = Xb.reset_index(drop=True).groupby([des.subject.to_numpy(), des.time.to_numpy()]).mean()
    blocks = [means.xs(t, level=1) for t in des.order]
    subjects = sorted(set.intersection(*(set(b.index) for b in blocks)))  # complete subjects only
    if len(subjects) < 2:
        raise ValueError("The Friedman test needs at least two subjects measured at every time point")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, pv = friedmanchisquare(*[b.loc[subjects].to_numpy() for b in blocks], axis=0)
    return {"time": np.asarray(pv, dtype=float)}


def run_mixed(p: MixedParams, session: Session, progress: Progress) -> MixedResult:
    des, d = design(session, p)
    need_subject(des, "A repeated-measures test")
    Xb = bin_matrix(d.X, p.bin_width).reset_index(drop=True)
    if p.test == "friedman":
        pvals = _friedman_pvalues(Xb, des)
    else:
        pvals = _mixed_pvalues(Xb.to_numpy(dtype=float), des, p.time_as == "numeric", progress)
    table = pd.DataFrame({"ppm": Xb.columns.to_numpy(dtype=float)})
    for term, pv in pvals.items():
        table[f"p_{term}"] = pv
        table[f"p_{term}_adj"] = _apply_correction(pv, None if p.correction == "none" else p.correction)
    rank = table[[c for c in table.columns if c.endswith("_adj")]].min(axis=1, skipna=True)
    table = table.iloc[np.argsort(rank.fillna(np.inf).to_numpy(), kind="stable")].reset_index(drop=True)
    feature = table["ppm"].iloc[0] if p.feature_ppm is None else p.feature_ppm
    progress(1.0)
    return MixedResult(table, Xb, des, float(feature), p.test)


P_FLOOR = 1e-16  # display only: below double precision, p-values are "zero"; keeps the plot's scale useful


def _short_p(v: float) -> str:
    return "n/a" if v != v else f"<{P_FLOOR:.0e}" if v < P_FLOOR else f"{v:.2g}"


def _column_at(r: MixedResult, ppm: Optional[float]) -> int:
    return int(np.abs(r.data.columns.to_numpy(dtype=float) - (r.feature if ppm is None else ppm)).argmin())


def _mixed_profile(r: MixedResult, style=None, ppm: Optional[float] = None) -> go.Figure:
    j = _column_at(r, ppm)
    at = float(r.data.columns[j])
    row = r.table.loc[np.isclose(r.table["ppm"], at)].iloc[0]
    notes = [f"{c[2:-4]} {_short_p(row[c])}" for c in r.table.columns if c.endswith("_adj")]
    title = f"{at:.4f} ppm<br><span style='font-size:0.8em'>adjusted p: {' · '.join(notes)}</span>"
    return profile_figure(r.data.iloc[:, j].to_numpy(), r.des, "mean", "sem", title, style)


def _pvalues(r: MixedResult, style=None, ppm: Optional[float] = None) -> go.Figure:
    t = r.table.sort_values("ppm")
    fig = go.Figure()
    for c in [c for c in t.columns if c.endswith("_adj")]:
        fig.add_trace(go.Scatter(x=t["ppm"], y=-np.log10(t[c].clip(lower=P_FLOOR)), name=c[2:-4], mode="lines+markers", marker={"size": 4}))
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="#888888")
    fig.add_vline(x=float(r.data.columns[_column_at(r, ppm)]), line_dash="dot", line_color="#888888")
    return fig.update_layout(
        title="Click a feature to show its time profile", template="plotly_white",
        xaxis={"autorange": "reversed", "title": "δ¹H (ppm)"}, yaxis_title="-log10 adjusted p",
    )


_common = {"section": SECTION, "requires": "import", "after": list(DATA_ORDER)}

LONGITUDINAL_STAGES: list[Stage] = [
    Stage(id="asca", title="ASCA", params=AscaParams, run=run_asca, table=lambda r: r.table,
          figures={"effects": _effects, "scores": _scores, "loadings": _loadings},
          **_common),
    Stage(id="mixedmodel", title="Mixed model", params=MixedParams, run=run_mixed, table=lambda r: r.table,
          figures={"profile": _mixed_profile, "pvalues": _pvalues}, companions={"profile": "pvalues"},
          ppm_views={"pvalues": "x"}, **_common),
]
