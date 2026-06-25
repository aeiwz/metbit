"""Train/test split and cross-validation utilities for metbit."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneOut,
    LeavePOut,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    learning_curve,
    train_test_split,
)

# ---------------------------------------------------------------------------
# CV strategy registry
# ---------------------------------------------------------------------------

_CV_STRATEGIES: Dict[str, str] = {
    "kfold":             "K-Fold",
    "stratified_kfold":  "Stratified K-Fold",
    "loo":               "Leave-One-Out",
    "leave_p_out":       "Leave-P-Out",
    "repeated_kfold":    "Repeated Stratified K-Fold",
    "shuffle_split":     "Monte Carlo (ShuffleSplit)",
    "group_kfold":       "Group K-Fold",
    "time_series":       "Time Series Split",
}


def _make_splitter(strategy: str, n_splits: int, **kwargs) -> Any:
    if strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=True,
                     random_state=kwargs.get("random_state", 42))
    if strategy == "stratified_kfold":
        return StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=kwargs.get("random_state", 42))
    if strategy == "loo":
        return LeaveOneOut()
    if strategy == "leave_p_out":
        p = kwargs.get("p", 2)
        return LeavePOut(p=p)
    if strategy == "repeated_kfold":
        n_repeats = kwargs.get("n_repeats", 5)
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=kwargs.get("random_state", 42))
    if strategy == "shuffle_split":
        test_size = kwargs.get("test_size", 0.2)
        return ShuffleSplit(n_splits=n_splits, test_size=test_size,
                            random_state=kwargs.get("random_state", 42))
    if strategy == "group_kfold":
        return GroupKFold(n_splits=n_splits)
    if strategy == "time_series":
        return TimeSeriesSplit(n_splits=n_splits)
    raise ValueError(
        f"Unknown cv_strategy {strategy!r}. "
        f"Choose from: {list(_CV_STRATEGIES.keys())}"
    )


# ---------------------------------------------------------------------------
# TrainTestSplit
# ---------------------------------------------------------------------------

class TrainTestSplit:
    """Simple stratified train/test holdout split with Plotly diagnostics.

    Args:
        X: Feature matrix (DataFrame or ndarray).
        y: Target labels (Series, ndarray, or list).
        test_size: Fraction of samples in the test set.
        stratify: Whether to stratify by y (keeps class proportions).
        random_state: Random seed for reproducibility.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.validation.splitter import TrainTestSplit
        >>> X = pd.DataFrame(np.random.rand(100, 20))
        >>> y = pd.Series(["A"] * 50 + ["B"] * 50)
        >>> tts = TrainTestSplit(X, y, test_size=0.2)
        >>> X_train, X_test, y_train, y_test = tts.split()
        >>> fig = tts.plot_split()
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List[Any]],
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: int = 42,
    ) -> None:
        self.X = X
        self.y = pd.Series(y).reset_index(drop=True)
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state
        self._X_train = self._X_test = self._y_train = self._y_test = None

    # ------------------------------------------------------------------
    def split(self):
        """Perform the split and return (X_train, X_test, y_train, y_test).

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).

        Examples:
            >>> X_train, X_test, y_train, y_test = tts.split()
        """
        strat = self.y if self.stratify else None
        self._X_train, self._X_test, self._y_train, self._y_test = (
            train_test_split(
                self.X, self.y,
                test_size=self.test_size,
                stratify=strat,
                random_state=self.random_state,
            )
        )
        return self._X_train, self._X_test, self._y_train, self._y_test

    def get_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising train/test class distributions.

        Returns:
            DataFrame with columns: class, train_n, test_n, train_pct, test_pct.

        Examples:
            >>> summary = tts.get_summary()
        """
        if self._y_train is None:
            self.split()
        classes = sorted(self.y.unique())
        rows = []
        for cls in classes:
            tr_n = int((self._y_train == cls).sum())
            te_n = int((self._y_test == cls).sum())
            rows.append({
                "class":     cls,
                "train_n":   tr_n,
                "test_n":    te_n,
                "train_pct": round(100 * tr_n / len(self._y_train), 1),
                "test_pct":  round(100 * te_n / len(self._y_test), 1),
            })
        return pd.DataFrame(rows)

    def plot_split(
        self,
        fig_height: int = 400,
        fig_width: int = 700,
        font_size: int = 13,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Bar chart showing class distribution in train and test sets.

        Args:
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Base font size.
            title: Optional plot title.

        Returns:
            go.Figure

        Examples:
            >>> fig = tts.plot_split()
            >>> fig.show()
        """
        summary = self.get_summary()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Train", x=summary["class"].astype(str),
            y=summary["train_n"], marker_color="#2563eb",
            text=summary["train_pct"].apply(lambda v: f"{v}%"),
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="Test", x=summary["class"].astype(str),
            y=summary["test_n"], marker_color="#f59e0b",
            text=summary["test_pct"].apply(lambda v: f"{v}%"),
            textposition="outside",
        ))
        n_tr, n_te = len(self._y_train), len(self._y_test)
        fig.update_layout(
            title=title or f"Train / Test Split  (train={n_tr}, test={n_te}, test_size={self.test_size})",
            barmode="group",
            xaxis_title="Class",
            yaxis_title="Number of samples",
            height=fig_height, width=fig_width,
            font=dict(size=font_size),
            legend=dict(orientation="h", y=1.1),
        )
        return fig


# ---------------------------------------------------------------------------
# CrossValidator
# ---------------------------------------------------------------------------

class CrossValidator:
    """Unified cross-validation with multiple splitting strategies and Plotly output.

    Supports any sklearn-compatible estimator.

    Args:
        estimator: sklearn-compatible classifier.
        X: Feature matrix (DataFrame or ndarray).
        y: Target labels (Series, ndarray, or list).
        cv_strategy: Splitting strategy. One of:

            - ``"kfold"`` – K-Fold (shuffled)
            - ``"stratified_kfold"`` – Stratified K-Fold (default)
            - ``"loo"`` – Leave-One-Out
            - ``"leave_p_out"`` – Leave-P-Out
            - ``"repeated_kfold"`` – Repeated Stratified K-Fold
            - ``"shuffle_split"`` – Monte Carlo (ShuffleSplit)
            - ``"group_kfold"`` – Group K-Fold (requires ``groups``)
            - ``"time_series"`` – Time Series Split

        n_splits: Number of CV folds (ignored for LOO).
        scoring: Scoring metric string (sklearn convention). Default ``"balanced_accuracy"``.
        random_state: Random seed.
        n_jobs: Parallel jobs for cross_val_score (-1 = all cores).
        groups: Sample group labels for GroupKFold.
        n_repeats: Number of repeats for ``"repeated_kfold"``.
        p: P value for ``"leave_p_out"``.
        test_size: Test fraction for ``"shuffle_split"``.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from metbit.validation.splitter import CrossValidator
        >>> X = pd.DataFrame(np.random.rand(80, 20))
        >>> y = pd.Series(["A"] * 40 + ["B"] * 40)
        >>> cv = CrossValidator(RandomForestClassifier(n_estimators=10), X, y)
        >>> cv.fit()
        >>> fig = cv.plot_scores()
        >>> summary = cv.get_summary()
    """

    def __init__(
        self,
        estimator: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List[Any]],
        cv_strategy: str = "stratified_kfold",
        n_splits: int = 5,
        scoring: str = "balanced_accuracy",
        random_state: int = 42,
        n_jobs: int = -1,
        groups: Optional[Union[np.ndarray, List]] = None,
        n_repeats: int = 5,
        p: int = 2,
        test_size: float = 0.2,
    ) -> None:
        if cv_strategy not in _CV_STRATEGIES:
            raise ValueError(
                f"cv_strategy must be one of {list(_CV_STRATEGIES.keys())}, "
                f"got {cv_strategy!r}"
            )
        self.estimator = estimator
        self.X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X.values
        self.y = np.asarray(y)
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.groups = groups
        self._splitter = _make_splitter(
            cv_strategy, n_splits,
            random_state=random_state,
            n_repeats=n_repeats,
            p=p,
            test_size=test_size,
        )
        self.scores_: Optional[np.ndarray] = None
        self._strategy_label = _CV_STRATEGIES[cv_strategy]

    # ------------------------------------------------------------------
    def fit(self) -> "CrossValidator":
        """Run cross-validation and store per-fold scores.

        Returns:
            self

        Examples:
            >>> cv.fit()
            >>> print(cv.scores_)
        """
        self.scores_ = cross_val_score(
            self.estimator, self.X, self.y,
            cv=self._splitter,
            scoring=self.scoring,
            groups=self.groups,
            n_jobs=self.n_jobs,
        )
        return self

    def get_scores(self) -> pd.DataFrame:
        """Return per-fold scores as a DataFrame.

        Returns:
            DataFrame with columns: fold, score.

        Examples:
            >>> df = cv.get_scores()
        """
        self._check_fitted()
        return pd.DataFrame({
            "fold":  np.arange(1, len(self.scores_) + 1),
            "score": self.scores_,
        })

    def get_summary(self) -> pd.DataFrame:
        """Return mean, std, min, max of CV scores.

        Returns:
            Single-row DataFrame with summary statistics.

        Examples:
            >>> summary = cv.get_summary()
        """
        self._check_fitted()
        return pd.DataFrame([{
            "strategy": self._strategy_label,
            "scoring":  self.scoring,
            "n_folds":  len(self.scores_),
            "mean":     round(float(self.scores_.mean()), 4),
            "std":      round(float(self.scores_.std()), 4),
            "min":      round(float(self.scores_.min()), 4),
            "max":      round(float(self.scores_.max()), 4),
        }])

    def plot_scores(
        self,
        fig_height: int = 420,
        fig_width: int = 700,
        font_size: int = 13,
        title: Optional[str] = None,
        color: str = "#2563eb",
    ) -> go.Figure:
        """Bar chart of per-fold CV scores with mean ± std annotation.

        Args:
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Base font size.
            title: Optional plot title.
            color: Bar color.

        Returns:
            go.Figure

        Examples:
            >>> fig = cv.plot_scores()
            >>> fig.show()
        """
        self._check_fitted()
        folds = np.arange(1, len(self.scores_) + 1)
        mean, std = self.scores_.mean(), self.scores_.std()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=folds, y=self.scores_,
            name=self.scoring, marker_color=color,
            text=[f"{s:.3f}" for s in self.scores_],
            textposition="outside",
        ))
        fig.add_hline(
            y=mean, line_dash="dash", line_color="#ef4444",
            annotation_text=f"Mean={mean:.3f} ±{std:.3f}",
            annotation_position="top right",
        )
        fig.update_layout(
            title=title or f"{self._strategy_label} — {self.scoring}",
            xaxis_title="Fold",
            yaxis_title=self.scoring,
            height=fig_height, width=fig_width,
            font=dict(size=font_size),
            showlegend=False,
        )
        return fig

    def plot_score_distribution(
        self,
        fig_height: int = 400,
        fig_width: int = 500,
        font_size: int = 13,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Box + strip plot of CV score distribution.

        Args:
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Base font size.
            title: Optional plot title.

        Returns:
            go.Figure

        Examples:
            >>> fig = cv.plot_score_distribution()
            >>> fig.show()
        """
        self._check_fitted()
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=self.scores_, name=self._strategy_label,
            boxpoints="all", jitter=0.3, pointpos=-1.5,
            marker_color="#2563eb",
            line_color="#1d4ed8",
        ))
        fig.update_layout(
            title=title or f"Score distribution — {self.scoring}",
            yaxis_title=self.scoring,
            height=fig_height, width=fig_width,
            font=dict(size=font_size),
            showlegend=False,
        )
        return fig

    def plot_learning_curve(
        self,
        train_sizes: Optional[np.ndarray] = None,
        fig_height: int = 450,
        fig_width: int = 750,
        font_size: int = 13,
        title: Optional[str] = None,
        n_jobs: int = -1,
    ) -> go.Figure:
        """Learning curve: training size vs. train and CV score.

        Args:
            train_sizes: Array of training set fractions. Defaults to
                ``np.linspace(0.1, 1.0, 8)``.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Base font size.
            title: Optional plot title.
            n_jobs: Parallel jobs.

        Returns:
            go.Figure

        Examples:
            >>> fig = cv.plot_learning_curve()
            >>> fig.show()
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 8)

        sizes, tr_scores, cv_scores = learning_curve(
            self.estimator, self.X, self.y,
            train_sizes=train_sizes,
            cv=self._splitter,
            scoring=self.scoring,
            groups=self.groups,
            n_jobs=n_jobs,
        )

        tr_mean, tr_std = tr_scores.mean(axis=1), tr_scores.std(axis=1)
        cv_mean, cv_std = cv_scores.mean(axis=1), cv_scores.std(axis=1)

        fig = go.Figure()
        # Training score
        fig.add_trace(go.Scatter(
            x=sizes, y=tr_mean, mode="lines+markers",
            name="Train score", line=dict(color="#2563eb"),
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([sizes, sizes[::-1]]),
            y=np.concatenate([tr_mean + tr_std, (tr_mean - tr_std)[::-1]]),
            fill="toself", fillcolor="rgba(37,99,235,0.15)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        # CV score
        fig.add_trace(go.Scatter(
            x=sizes, y=cv_mean, mode="lines+markers",
            name="CV score", line=dict(color="#10b981"),
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([sizes, sizes[::-1]]),
            y=np.concatenate([cv_mean + cv_std, (cv_mean - cv_std)[::-1]]),
            fill="toself", fillcolor="rgba(16,185,129,0.15)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig.update_layout(
            title=title or f"Learning Curve — {self.scoring}",
            xaxis_title="Training samples",
            yaxis_title=self.scoring,
            height=fig_height, width=fig_width,
            font=dict(size=font_size),
        )
        return fig

    def compare_strategies(
        self,
        strategies: Optional[List[str]] = None,
        fig_height: int = 450,
        fig_width: int = 900,
        font_size: int = 13,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Run and compare multiple CV strategies side-by-side.

        Runs the same estimator under each strategy (using this instance's
        n_splits and random_state) and returns a grouped box plot.

        Args:
            strategies: List of strategy names to compare. Defaults to
                ``["kfold", "stratified_kfold", "shuffle_split", "repeated_kfold"]``.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Base font size.
            title: Optional plot title.

        Returns:
            go.Figure

        Examples:
            >>> fig = cv.compare_strategies()
            >>> fig.show()
        """
        if strategies is None:
            strategies = ["kfold", "stratified_kfold", "shuffle_split", "repeated_kfold"]

        fig = go.Figure()
        palette = ["#2563eb", "#10b981", "#f59e0b", "#ef4444",
                   "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"]
        for i, strat in enumerate(strategies):
            splitter = _make_splitter(strat, self.n_splits,
                                      random_state=self.random_state)
            sc = cross_val_score(
                self.estimator, self.X, self.y,
                cv=splitter, scoring=self.scoring,
                groups=self.groups, n_jobs=self.n_jobs,
            )
            fig.add_trace(go.Box(
                y=sc, name=_CV_STRATEGIES[strat],
                boxpoints="all", jitter=0.3,
                marker_color=palette[i % len(palette)],
            ))
        fig.update_layout(
            title=title or f"CV Strategy Comparison — {self.scoring}",
            yaxis_title=self.scoring,
            height=fig_height, width=fig_width,
            font=dict(size=font_size),
        )
        return fig

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self.scores_ is None:
            raise RuntimeError("Call fit() before accessing results or plots.")


# ---------------------------------------------------------------------------
# available_strategies helper
# ---------------------------------------------------------------------------

def available_cv_strategies() -> pd.DataFrame:
    """Return a DataFrame listing all supported CV strategies.

    Returns:
        DataFrame with columns: key, name.

    Examples:
        >>> from metbit.validation.splitter import available_cv_strategies
        >>> print(available_cv_strategies())
    """
    return pd.DataFrame(
        [{"key": k, "name": v} for k, v in _CV_STRATEGIES.items()]
    )
