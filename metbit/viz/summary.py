from __future__ import annotations

# -*- coding: utf-8 -*-

__author__ = 'aeiwz'
__copyright__ = "Copyright 2024, Theerayut"

__license__ = "MIT"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Development"

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import plotly.graph_objects as go


class FeatureHeatmap:
    """Clustered heatmap of feature intensities across samples or groups.

    Visualises a sample-by-feature intensity matrix as an annotated heatmap
    with optional hierarchical clustering of both rows and columns. A group
    colour-bar can be overlaid on the sample axis when group labels are
    supplied.

    Args:
        df: DataFrame with rows=samples and columns=features.
        label: Group labels aligned with the rows of *df*. Used to draw a
            colour-bar annotation on top of the heatmap. Accepts a
            ``pd.Series`` or any list-like of the same length as ``df``.
        features: Explicit subset of column names to include. When ``None``
            all columns are used (subject to ``n_features`` in
            :meth:`plot`).
        scaling: Pre-processing applied to each feature column before
            display. One of ``"zscore"`` (zero mean, unit variance),
            ``"minmax"`` (scale to [0, 1]), or ``"none"`` (raw values).

    Raises:
        ValueError: If *scaling* is not one of the accepted values.
        ValueError: If *label* length does not match the number of rows in
            *df*.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from metbit.viz.summary import FeatureHeatmap
        >>> X = pd.DataFrame(
        ...     np.random.rand(40, 20),
        ...     columns=[f"f{i}" for i in range(20)],
        ... )
        >>> label = pd.Series(["A"] * 20 + ["B"] * 20)
        >>> hm = FeatureHeatmap(X, label=label, scaling="zscore")
        >>> fig = hm.plot(n_features=20)
        >>> isinstance(fig, go.Figure)
        True
    """

    _VALID_SCALING = {"zscore", "minmax", "none"}

    def __init__(
        self,
        df: pd.DataFrame,
        label: Optional[Union[pd.Series, list]] = None,
        features: Optional[List[str]] = None,
        scaling: str = "zscore",
    ) -> None:
        if scaling not in self._VALID_SCALING:
            raise ValueError(
                f"scaling must be one of {self._VALID_SCALING}, got {scaling!r}."
            )

        self._df_raw = df.copy()
        self.scaling = scaling

        if label is not None:
            label = pd.Series(label, name="group").reset_index(drop=True)
            if len(label) != len(df):
                raise ValueError(
                    f"label length ({len(label)}) must match number of rows in "
                    f"df ({len(df)})."
                )
        self.label = label

        if features is not None:
            missing = set(features) - set(df.columns)
            if missing:
                raise ValueError(f"Features not found in df: {missing}")
            self._df_raw = self._df_raw[features]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the chosen scaling to each column."""
        if self.scaling == "zscore":
            std = data.std(ddof=1)
            std = std.replace(0, 1)
            return (data - data.mean()) / std
        if self.scaling == "minmax":
            mn = data.min()
            mx = data.max()
            rng = (mx - mn).replace(0, 1)
            return (data - mn) / rng
        return data.copy()

    @staticmethod
    def _cluster_order(matrix: np.ndarray) -> np.ndarray:
        """Return row indices sorted by hierarchical clustering (Ward)."""
        if matrix.shape[0] < 2:
            return np.arange(matrix.shape[0])
        dist = pdist(matrix, metric="euclidean")
        Z = linkage(dist, method="ward")
        return leaves_list(Z)

    @staticmethod
    def _group_colormap(groups: pd.Series) -> dict:
        """Map unique group labels to distinct Plotly colours."""
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        unique = groups.unique().tolist()
        return {g: palette[i % len(palette)] for i, g in enumerate(unique)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_top_features(self, n: int = 50) -> pd.DataFrame:
        """Return the top *n* features ranked by across-sample variance.

        Args:
            n: Number of features to return. Capped at the total number of
                available features.

        Returns:
            DataFrame with columns ``feature`` and ``variance``, sorted
            descending by variance.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from metbit.viz.summary import FeatureHeatmap
            >>> X = pd.DataFrame(np.random.rand(20, 10),
            ...                  columns=[f"f{i}" for i in range(10)])
            >>> hm = FeatureHeatmap(X)
            >>> top = hm.get_top_features(n=5)
            >>> list(top.columns)
            ['feature', 'variance']
            >>> len(top)
            5
        """
        n = min(n, self._df_raw.shape[1])
        variances = self._df_raw.var(ddof=1).sort_values(ascending=False)
        return pd.DataFrame(
            {"feature": variances.index[:n], "variance": variances.values[:n]}
        ).reset_index(drop=True)

    def plot(
        self,
        n_features: int = 50,
        cluster_samples: bool = True,
        cluster_features: bool = True,
        colorscale: str = "RdBu_r",
        fig_height: int = 800,
        fig_width: int = 1000,
        font_size: int = 11,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Render the clustered heatmap.

        Args:
            n_features: Maximum number of features to display. The top
                features by variance are selected automatically.
            cluster_samples: Whether to reorder samples (rows) by
                hierarchical clustering.
            cluster_features: Whether to reorder features (columns) by
                hierarchical clustering.
            colorscale: Any Plotly-compatible diverging colorscale name,
                e.g. ``"RdBu_r"``, ``"Viridis"``, or ``"RdYlGn"``.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Base font size for axis labels and ticks.
            title: Optional figure title. Defaults to
                ``"Feature Heatmap"``.

        Returns:
            A :class:`plotly.graph_objects.Figure` ready for display or
            export.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from metbit.viz.summary import FeatureHeatmap
            >>> X = pd.DataFrame(np.random.rand(40, 20),
            ...                  columns=[f"f{i}" for i in range(20)])
            >>> label = pd.Series(["A"] * 20 + ["B"] * 20)
            >>> hm = FeatureHeatmap(X, label=label)
            >>> fig = hm.plot(n_features=15)
            >>> isinstance(fig, go.Figure)
            True
        """
        # Select features
        top_feats = self.get_top_features(n=n_features)["feature"].tolist()
        data = self._df_raw[top_feats].copy()

        # Scale
        scaled = self._scale(data)

        # Cluster
        sample_idx = np.arange(scaled.shape[0])
        feat_idx = np.arange(scaled.shape[1])

        if cluster_samples and scaled.shape[0] > 1:
            sample_idx = self._cluster_order(scaled.values)
        if cluster_features and scaled.shape[1] > 1:
            feat_idx = self._cluster_order(scaled.values.T)

        matrix = scaled.values[np.ix_(sample_idx, feat_idx)]
        sample_names = scaled.index[sample_idx].astype(str).tolist()
        feat_names = scaled.columns[feat_idx].astype(str).tolist()

        # Truncate feature labels when there are many
        max_label_len = 20
        feat_labels = [
            f[:max_label_len] + "..." if len(f) > max_label_len else f
            for f in feat_names
        ]

        traces: List[go.BaseTraceType] = []

        # Group colour-bar annotation on top
        if self.label is not None:
            label_ordered = self.label.reset_index(drop=True).iloc[sample_idx]
            cmap = self._group_colormap(label_ordered)
            bar_colors = [cmap[g] for g in label_ordered]

            # Thin scatter trace that acts as a visual colour stripe
            traces.append(
                go.Bar(
                    x=list(range(len(sample_names))),
                    y=[1] * len(sample_names),
                    marker_color=bar_colors,
                    showlegend=True,
                    name="",
                    hovertemplate="%{text}<extra></extra>",
                    text=label_ordered.tolist(),
                    xaxis="x",
                    yaxis="y2",
                )
            )
            # Legend entries for each group
            for grp, clr in cmap.items():
                traces.append(
                    go.Bar(
                        x=[None],
                        y=[None],
                        marker_color=clr,
                        name=str(grp),
                        showlegend=True,
                        xaxis="x",
                        yaxis="y2",
                    )
                )

        # Main heatmap
        traces.append(
            go.Heatmap(
                z=matrix.T.tolist(),
                x=sample_names,
                y=feat_labels,
                colorscale=colorscale,
                zmid=0,
                colorbar=dict(title="Scaled intensity", thickness=15),
                hovertemplate=(
                    "Sample: %{x}<br>Feature: %{y}<br>Value: %{z:.3f}<extra></extra>"
                ),
                xaxis="x",
                yaxis="y",
            )
        )

        layout_kwargs: dict = dict(
            title=dict(text=title or "Feature Heatmap", font_size=font_size + 4),
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            xaxis=dict(
                showticklabels=scaled.shape[0] <= 60,
                tickfont=dict(size=max(7, font_size - 2)),
            ),
            yaxis=dict(
                tickfont=dict(size=max(7, font_size - 2)),
                autorange="reversed",
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        if self.label is not None:
            layout_kwargs["yaxis2"] = dict(
                domain=[0.92, 1.0],
                showticklabels=False,
                showgrid=False,
            )
            layout_kwargs["yaxis"]["domain"] = [0.0, 0.90]
            layout_kwargs["barmode"] = "stack"

        fig = go.Figure(data=traces, layout=go.Layout(**layout_kwargs))
        return fig


# ---------------------------------------------------------------------------


class CorrelationMatrix:
    """Pairwise feature or sample correlation heatmap.

    Computes a Pearson or Spearman correlation matrix and renders it as an
    interactive Plotly heatmap, with optional hierarchical clustering to
    group correlated entities together.

    Args:
        df: DataFrame with rows=samples and columns=features.
        method: Correlation method. One of ``"pearson"`` or
            ``"spearman"``.

    Raises:
        ValueError: If *method* is not ``"pearson"`` or ``"spearman"``.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from metbit.viz.summary import CorrelationMatrix
        >>> X = pd.DataFrame(np.random.rand(30, 15),
        ...                  columns=[f"f{i}" for i in range(15)])
        >>> cm = CorrelationMatrix(X)
        >>> fig = cm.plot_features(n_features=10)
        >>> isinstance(fig, go.Figure)
        True
    """

    _VALID_METHODS = {"pearson", "spearman"}

    def __init__(self, df: pd.DataFrame, method: str = "pearson") -> None:
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"method must be one of {self._VALID_METHODS}, got {method!r}."
            )
        self._df = df.copy()
        self.method = method

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _top_features(self, n: int) -> List[str]:
        n = min(n, self._df.shape[1])
        return (
            self._df.var(ddof=1)
            .sort_values(ascending=False)
            .index[:n]
            .tolist()
        )

    @staticmethod
    def _cluster_order(matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[0] < 2:
            return np.arange(matrix.shape[0])  # pragma: no cover
        dist = pdist(matrix, metric="euclidean")
        Z = linkage(dist, method="ward")
        return leaves_list(Z)

    @staticmethod
    def _group_colormap(groups: pd.Series) -> dict:
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        unique = groups.unique().tolist()
        return {g: palette[i % len(palette)] for i, g in enumerate(unique)}

    def _build_heatmap_fig(
        self,
        corr: pd.DataFrame,
        labels: List[str],
        title: str,
        colorscale: str,
        fig_height: int,
        fig_width: int,
        font_size: int,
        group_colors: Optional[List[str]] = None,
        group_names: Optional[List[str]] = None,
    ) -> go.Figure:
        """Shared figure-building logic for both plot methods."""
        z = corr.values.tolist()

        # Mask diagonal by setting it to None so it renders in a neutral shade
        z_masked = [row[:] for row in z]
        for i in range(len(z_masked)):
            z_masked[i][i] = None

        customdata = corr.values.tolist()

        traces: List[go.BaseTraceType] = [
            go.Heatmap(
                z=z_masked,
                x=labels,
                y=labels,
                zmin=-1,
                zmax=1,
                zmid=0,
                colorscale=colorscale,
                colorbar=dict(title="r", thickness=15),
                customdata=customdata,
                hovertemplate=(
                    "x: %{x}<br>y: %{y}<br>r: %{customdata:.3f}<extra></extra>"
                ),
                xaxis="x",
                yaxis="y",
            )
        ]

        layout_kwargs: dict = dict(
            title=dict(text=title, font_size=font_size + 4),
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            xaxis=dict(tickfont=dict(size=max(7, font_size - 2)), side="bottom"),
            yaxis=dict(
                tickfont=dict(size=max(7, font_size - 2)), autorange="reversed"
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        if group_colors is not None and group_names is not None:
            traces.append(
                go.Bar(
                    x=labels,
                    y=[1] * len(labels),
                    marker_color=group_colors,
                    showlegend=False,
                    hovertemplate="%{text}<extra></extra>",
                    text=group_names,
                    xaxis="x",
                    yaxis="y2",
                )
            )
            layout_kwargs["yaxis2"] = dict(
                domain=[0.92, 1.0],
                showticklabels=False,
                showgrid=False,
            )
            layout_kwargs["yaxis"]["domain"] = [0.0, 0.90]
            layout_kwargs["barmode"] = "stack"

        return go.Figure(data=traces, layout=go.Layout(**layout_kwargs))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_correlation_matrix(self, n_features: int = 30) -> pd.DataFrame:
        """Return the raw feature-feature correlation matrix.

        Args:
            n_features: Number of features (selected by highest variance) to
                include in the correlation matrix.

        Returns:
            Square DataFrame of shape ``(n_features, n_features)`` containing
            pairwise correlation coefficients.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from metbit.viz.summary import CorrelationMatrix
            >>> X = pd.DataFrame(np.random.rand(20, 10),
            ...                  columns=[f"f{i}" for i in range(10)])
            >>> cm = CorrelationMatrix(X)
            >>> corr = cm.get_correlation_matrix(n_features=5)
            >>> corr.shape
            (5, 5)
        """
        feats = self._top_features(n_features)
        subset = self._df[feats]
        return subset.corr(method=self.method)

    def plot_features(
        self,
        n_features: int = 30,
        cluster: bool = True,
        colorscale: str = "RdBu_r",
        fig_height: int = 700,
        fig_width: int = 750,
        font_size: int = 10,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Plot a feature-by-feature correlation heatmap.

        The top *n_features* features by variance are selected and their
        pairwise correlations displayed. The diagonal is masked to grey to
        avoid visual distraction from the trivial self-correlation of 1.

        Args:
            n_features: Number of features to include.
            cluster: Reorder features by hierarchical clustering when
                ``True``.
            colorscale: Diverging Plotly colorscale for correlation values.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Base font size.
            title: Optional figure title.

        Returns:
            :class:`plotly.graph_objects.Figure`

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from metbit.viz.summary import CorrelationMatrix
            >>> X = pd.DataFrame(np.random.rand(30, 15),
            ...                  columns=[f"f{i}" for i in range(15)])
            >>> cm = CorrelationMatrix(X)
            >>> fig = cm.plot_features(n_features=10, cluster=False)
            >>> isinstance(fig, go.Figure)
            True
        """
        corr = self.get_correlation_matrix(n_features=n_features)

        if cluster and corr.shape[0] > 1:
            order = self._cluster_order(corr.values)
            corr = corr.iloc[order, :].iloc[:, order]

        labels = [
            f[:20] + "..." if len(f) > 20 else f for f in corr.columns.tolist()
        ]

        return self._build_heatmap_fig(
            corr=corr,
            labels=labels,
            title=title or f"Feature Correlation Matrix ({self.method.capitalize()})",
            colorscale=colorscale,
            fig_height=fig_height,
            fig_width=fig_width,
            font_size=font_size,
        )

    def plot_samples(
        self,
        cluster: bool = True,
        label: Optional[Union[pd.Series, list]] = None,
        colorscale: str = "RdBu_r",
        fig_height: int = 700,
        fig_width: int = 750,
        font_size: int = 10,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Plot a sample-by-sample correlation heatmap.

        Args:
            cluster: Reorder samples by hierarchical clustering when
                ``True``.
            label: Optional group labels for samples. When provided, a
                colour-coded stripe is rendered above the heatmap.
            colorscale: Diverging Plotly colorscale.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Base font size.
            title: Optional figure title.

        Returns:
            :class:`plotly.graph_objects.Figure`

        Raises:
            ValueError: If *label* length does not match the number of
                samples in *df*.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from metbit.viz.summary import CorrelationMatrix
            >>> X = pd.DataFrame(np.random.rand(20, 10),
            ...                  columns=[f"f{i}" for i in range(10)])
            >>> lbl = pd.Series(["A"] * 10 + ["B"] * 10)
            >>> cm = CorrelationMatrix(X)
            >>> fig = cm.plot_samples(label=lbl)
            >>> isinstance(fig, go.Figure)
            True
        """
        if label is not None:
            label = pd.Series(label, name="group").reset_index(drop=True)
            if len(label) != len(self._df):
                raise ValueError(
                    f"label length ({len(label)}) must match number of rows in "
                    f"df ({len(self._df)})."
                )

        corr = self._df.T.corr(method=self.method)

        if cluster and corr.shape[0] > 1:
            order = self._cluster_order(corr.values)
            corr = corr.iloc[order, :].iloc[:, order]
            sample_order = order
        else:
            sample_order = np.arange(len(self._df))

        sample_names = self._df.index[sample_order].astype(str).tolist()

        group_colors = None
        group_names = None
        if label is not None:
            label_ordered = label.iloc[sample_order]
            cmap = self._group_colormap(label_ordered)
            group_colors = [cmap[g] for g in label_ordered]
            group_names = label_ordered.tolist()

        return self._build_heatmap_fig(
            corr=corr,
            labels=sample_names,
            title=title or f"Sample Correlation Matrix ({self.method.capitalize()})",
            colorscale=colorscale,
            fig_height=fig_height,
            fig_width=fig_width,
            font_size=font_size,
            group_colors=group_colors,
            group_names=group_names,
        )


# ---------------------------------------------------------------------------


class PValueTable:
    """Visual table of pairwise statistical test results with significance stars.

    Accepts a tidy DataFrame and performs univariate statistical tests for
    one or more numeric columns, comparing values across groups defined by
    *group_col*. The results are presented as a colour-coded Plotly table.

    Args:
        df: Tidy DataFrame containing at least *group_col* and one or more
            numeric value columns.
        group_col: Name of the column that encodes group membership.
        value_col: Single numeric column to test. When ``None`` every
            numeric column (excluding *group_col*) is tested.
        test: Statistical test to apply. ``"auto"`` selects a t-test for
            two-group comparisons and one-way ANOVA for three or more.
            Explicit choices: ``"ttest"``, ``"mannwhitney"``,
            ``"anova"``, ``"kruskal"``.
        correct_p: Multiple-testing correction method accepted by
            :func:`statsmodels.stats.multitest.multipletests`, e.g.
            ``"fdr_bh"``, ``"bonferroni"``, or ``None`` to skip
            correction.
        p_threshold: Significance threshold for colouring. Default
            ``0.05``.

    Raises:
        ValueError: If *group_col* is not present in *df*.
        ValueError: If *test* is not one of the accepted values.
        ValueError: If *value_col* is specified but not found in *df*.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from metbit.viz.summary import PValueTable
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({
        ...     "group": ["A"] * 20 + ["B"] * 20,
        ...     "glucose": rng.normal(5, 1, 40),
        ...     "lactate": rng.normal(2, 0.5, 40),
        ... })
        >>> pv = PValueTable(df, group_col="group")
        >>> tbl = pv.get_table()
        >>> list(tbl.columns)
        ['feature', 'statistic', 'p_value', 'p_adj', 'stars']
    """

    _VALID_TESTS = {"auto", "ttest", "mannwhitney", "anova", "kruskal"}

    def __init__(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: Optional[str] = None,
        test: str = "auto",
        correct_p: Optional[str] = "fdr_bh",
        p_threshold: float = 0.05,
    ) -> None:
        if group_col not in df.columns:
            raise ValueError(f"group_col {group_col!r} not found in df.")
        if test not in self._VALID_TESTS:
            raise ValueError(
                f"test must be one of {self._VALID_TESTS}, got {test!r}."
            )
        if value_col is not None and value_col not in df.columns:
            raise ValueError(f"value_col {value_col!r} not found in df.")

        self._df = df.copy()
        self.group_col = group_col
        self.value_col = value_col
        self.test = test
        self.correct_p = correct_p
        self.p_threshold = p_threshold

        self._results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _numeric_columns(self) -> List[str]:
        if self.value_col is not None:
            return [self.value_col]
        return [
            c
            for c in self._df.select_dtypes(include="number").columns
            if c != self.group_col
        ]

    @staticmethod
    def _stars(p: float) -> str:
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    def _run_test(
        self, groups: List[np.ndarray], test_name: str
    ) -> tuple:
        """Return (statistic, p_value) for the chosen test."""
        from scipy import stats as sp_stats

        if test_name == "ttest":
            if len(groups) != 2:
                raise ValueError("ttest requires exactly 2 groups.")
            stat, p = sp_stats.ttest_ind(*groups, equal_var=False)
        elif test_name == "mannwhitney":
            if len(groups) != 2:
                raise ValueError("mannwhitney requires exactly 2 groups.")
            stat, p = sp_stats.mannwhitneyu(*groups, alternative="two-sided")
        elif test_name == "anova":
            stat, p = sp_stats.f_oneway(*groups)
        elif test_name == "kruskal":
            stat, p = sp_stats.kruskal(*groups)
        else:  # pragma: no cover  – validated at init; reached only by direct _run_test calls
            raise ValueError(f"Unknown test: {test_name!r}")
        return float(stat), float(p)

    def _resolve_test(self, n_groups: int) -> str:
        if self.test != "auto":
            return self.test
        return "ttest" if n_groups == 2 else "anova"

    def _compute(self) -> pd.DataFrame:
        cols = self._numeric_columns()
        groups_data = {
            g: self._df[self._df[self.group_col] == g]
            for g in self._df[self.group_col].unique()
        }
        group_keys = list(groups_data.keys())
        n_groups = len(group_keys)
        test_name = self._resolve_test(n_groups)

        rows = []
        for col in cols:
            group_arrays = [
                groups_data[g][col].dropna().values for g in group_keys
            ]
            try:
                stat, p = self._run_test(group_arrays, test_name)
            except Exception:
                stat, p = float("nan"), float("nan")
            rows.append({"feature": col, "statistic": stat, "p_value": p})

        result = pd.DataFrame(rows)

        # Multiple-testing correction
        if self.correct_p is not None and not result["p_value"].isna().all():
            from statsmodels.stats.multitest import multipletests

            mask = ~result["p_value"].isna()
            if mask.sum() > 0:
                _, p_adj, _, _ = multipletests(
                    result.loc[mask, "p_value"].values,
                    method=self.correct_p,
                )
                result.loc[mask, "p_adj"] = p_adj
            else:  # pragma: no cover
                result["p_adj"] = result["p_value"]
        else:
            result["p_adj"] = result["p_value"]

        result["stars"] = result["p_adj"].apply(
            lambda v: self._stars(v) if not np.isnan(v) else "nd"
        )
        return result[["feature", "statistic", "p_value", "p_adj", "stars"]]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_table(self) -> pd.DataFrame:
        """Return the statistical results as a tidy DataFrame.

        Computes the test results lazily on first call and caches them for
        subsequent calls.

        Returns:
            DataFrame with columns:

            - ``feature``: name of the tested column
            - ``statistic``: test statistic (F, t, or U depending on test)
            - ``p_value``: raw p-value
            - ``p_adj``: adjusted p-value (or raw if *correct_p* is
              ``None``)
            - ``stars``: significance annotation (``***``, ``**``,
              ``*``, or ``ns``)

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from metbit.viz.summary import PValueTable
            >>> rng = np.random.default_rng(42)
            >>> df = pd.DataFrame({
            ...     "group": ["A"] * 20 + ["B"] * 20,
            ...     "x": rng.normal(0, 1, 40),
            ... })
            >>> pv = PValueTable(df, group_col="group")
            >>> tbl = pv.get_table()
            >>> "p_value" in tbl.columns
            True
        """
        if self._results is None:
            self._results = self._compute()
        return self._results.copy()

    def plot(
        self,
        fig_height: int = 600,
        fig_width: int = 900,
        font_size: int = 12,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Render a colour-coded significance table.

        Each row corresponds to a tested feature. Cells in the adjusted
        p-value column are coloured green when the result is significant
        (``p_adj < p_threshold``) and grey otherwise. The full p-value and
        star annotation are shown in each row.

        Args:
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Font size for table cells and header.
            title: Optional figure title. Defaults to
                ``"Statistical Test Results"``.

        Returns:
            :class:`plotly.graph_objects.Figure`

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from metbit.viz.summary import PValueTable
            >>> rng = np.random.default_rng(0)
            >>> df = pd.DataFrame({
            ...     "group": ["A"] * 15 + ["B"] * 15,
            ...     "alanine": rng.normal(3, 1, 30),
            ...     "valine": rng.normal(2, 1, 30),
            ... })
            >>> pv = PValueTable(df, group_col="group")
            >>> fig = pv.plot()
            >>> isinstance(fig, go.Figure)
            True
        """
        tbl = self.get_table()

        # Build per-cell colours
        def _cell_color(p_adj: float) -> str:
            if np.isnan(p_adj):
                return "#f5f5f5"  # pragma: no cover
            return "#d4edda" if p_adj < self.p_threshold else "#e9ecef"

        p_adj_colors = [_cell_color(v) for v in tbl["p_adj"]]

        # Format numeric columns
        def _fmt(v: float, decimals: int = 4) -> str:
            if np.isnan(v):
                return "nd"  # pragma: no cover
            return f"{v:.{decimals}f}"

        columns = ["Feature", "Statistic", "p-value", "p-adj", "Sig."]
        cell_values = [
            tbl["feature"].tolist(),
            [_fmt(v, 4) for v in tbl["statistic"]],
            [_fmt(v, 4) for v in tbl["p_value"]],
            [_fmt(v, 4) for v in tbl["p_adj"]],
            tbl["stars"].tolist(),
        ]

        # Colour scheme: feature=white, statistic=white,
        # p_value=white, p_adj=significance colour, stars=same
        n = len(tbl)
        white_col = ["white"] * n
        cell_colors = [
            white_col,
            white_col,
            white_col,
            p_adj_colors,
            p_adj_colors,
        ]

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=[f"<b>{c}</b>" for c in columns],
                        fill_color="#343a40",
                        font=dict(color="white", size=font_size),
                        align="center",
                        height=32,
                    ),
                    cells=dict(
                        values=cell_values,
                        fill_color=cell_colors,
                        align=["left", "center", "center", "center", "center"],
                        font=dict(size=font_size - 1),
                        height=28,
                    ),
                )
            ],
            layout=go.Layout(
                title=dict(
                    text=title or "Statistical Test Results",
                    font_size=font_size + 4,
                ),
                height=fig_height,
                width=fig_width,
                paper_bgcolor="white",
            ),
        )
        return fig
