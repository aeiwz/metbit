# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "aeiwz"
__copyright__ = "Copyright 2024, Theerayut"
__license__ = "MIT"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Development"

import numpy as np
import pandas as pd
from scipy import stats

import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    GridSearchCV,
)
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix as sk_confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


class ModelValidator:
    """Wraps any sklearn-compatible estimator and provides comprehensive
    validation metrics and Plotly figures for NMR metabolomics workflows.

    Args:
        estimator: Any sklearn-compatible classifier that implements
            ``fit``, ``predict``, and ``predict_proba``.
        X: Feature matrix of shape (n_samples, n_features).
        y: Target labels of shape (n_samples,).
        cv: Number of stratified k-fold splits used for all CV-based
            methods. Defaults to 5.
        random_state: Random seed for reproducibility. Defaults to 42.
        class_names: Display names for each class. If None, the unique
            sorted values of ``y`` are used.

    Attributes:
        roc_results_ (dict): Populated by :meth:`roc_auc`. Maps each
            class label to a dict containing ``fpr``, ``tpr``, and
            ``auc``.
        bootstrap_scores_ (np.ndarray): Populated by
            :meth:`bootstrap_ci`. Array of per-iteration metric values.
        bootstrap_ci_ (tuple[float, float]): Populated by
            :meth:`bootstrap_ci`. Lower and upper confidence bounds.

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> import pandas as pd
        >>> X_arr, y_arr = make_classification(n_samples=100, random_state=0)
        >>> X = pd.DataFrame(X_arr)
        >>> validator = ModelValidator(
        ...     estimator=RandomForestClassifier(random_state=0),
        ...     X=X,
        ...     y=y_arr,
        ...     cv=3,
        ... )
        >>> fig = validator.roc_auc()
        >>> df = validator.get_cv_summary()
    """

    def __init__(
        self,
        estimator,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | list,
        cv: int = 5,
        random_state: int = 42,
        class_names: list[str] | None = None,
    ) -> None:
        self.estimator = estimator
        self.X = np.array(X)
        self.y = np.array(y)
        self.cv = cv
        self.random_state = random_state

        unique_classes = np.unique(self.y)
        if class_names is not None:
            if len(class_names) != len(unique_classes):
                raise ValueError(
                    f"class_names length ({len(class_names)}) must match "
                    f"number of unique classes ({len(unique_classes)})."
                )
            self.class_names = list(class_names)
        else:
            self.class_names = [str(c) for c in unique_classes]

        self._unique_classes = unique_classes
        self._n_classes = len(unique_classes)

        # Populated lazily by public methods
        self.roc_results_: dict = {}
        self.bootstrap_scores_: np.ndarray | None = None
        self.bootstrap_ci_: tuple[float, float] | None = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def roc_auc(
        self,
        fig_height: int = 600,
        fig_width: int = 800,
        font_size: int = 14,
    ) -> go.Figure:
        """Compute and plot ROC curves using stratified k-fold CV.

        One curve is produced per class (one-vs-rest) plus a
        micro-average curve. AUC values are annotated in the legend.
        Results are stored in ``self.roc_results_``.

        Args:
            fig_height: Height of the returned Plotly figure in pixels.
                Defaults to 600.
            fig_width: Width of the returned Plotly figure in pixels.
                Defaults to 800.
            font_size: Base font size for the figure. Defaults to 14.

        Returns:
            Plotly Figure containing the ROC curve plot.

        Examples:
            >>> fig = validator.roc_auc()
            >>> fig.show()
        """
        cv_splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        y_prob = cross_val_predict(
            self.estimator,
            self.X,
            self.y,
            cv=cv_splitter,
            method="predict_proba",
        )

        # Binarize labels for multi-class one-vs-rest
        y_bin = label_binarize(self.y, classes=self._unique_classes)
        if self._n_classes == 2:
            # label_binarize returns shape (n, 1) for binary; expand to (n, 2)
            y_bin = np.hstack([1 - y_bin, y_bin])

        self.roc_results_ = {}
        traces: list[go.Scatter] = []

        for idx, cls in enumerate(self._unique_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob[:, idx])
            roc_auc_val = auc(fpr, tpr)
            label = self.class_names[idx]
            self.roc_results_[label] = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc_val,
            }
            traces.append(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{label} (AUC = {roc_auc_val:.3f})",
                    line=dict(width=2),
                )
            )

        # Micro-average
        fpr_micro, tpr_micro, _ = roc_curve(
            y_bin.ravel(), y_prob.ravel()
        )
        auc_micro = auc(fpr_micro, tpr_micro)
        self.roc_results_["micro-average"] = {
            "fpr": fpr_micro,
            "tpr": tpr_micro,
            "auc": auc_micro,
        }
        traces.append(
            go.Scatter(
                x=fpr_micro,
                y=tpr_micro,
                mode="lines",
                name=f"Micro-average (AUC = {auc_micro:.3f})",
                line=dict(dash="dash", width=2, color="black"),
            )
        )

        # Random chance diagonal
        traces.append(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random chance",
                line=dict(dash="dot", width=1, color="grey"),
                showlegend=True,
            )
        )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title="ROC Curves (Stratified {}-Fold CV)".format(self.cv),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            legend=dict(x=0.6, y=0.05),
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.02]),
        )
        return fig

    def nested_cv(
        self,
        inner_cv: int = 3,
        param_grid: dict | None = None,
        scoring: str = "balanced_accuracy",
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Perform nested cross-validation.

        The outer loop uses ``self.cv`` stratified folds. When
        ``param_grid`` is supplied, the inner loop runs a
        ``GridSearchCV`` with ``inner_cv`` folds for hyperparameter
        selection. When ``param_grid`` is None, the inner loop is
        skipped and plain outer CV is performed.

        Args:
            inner_cv: Number of inner folds for hyperparameter search.
                Only used when ``param_grid`` is provided. Defaults to 3.
            param_grid: Parameter grid passed to ``GridSearchCV``.
                If None, no inner loop is performed. Defaults to None.
            scoring: Sklearn scoring string used for both inner and
                outer evaluation. Defaults to "balanced_accuracy".
            n_jobs: Number of parallel jobs. Defaults to -1 (all CPUs).

        Returns:
            DataFrame with columns ``fold``, ``train_score``, and
            ``test_score`` (one row per outer fold). Mean and std of
            test scores are printed to stdout.

        Examples:
            >>> df = validator.nested_cv(param_grid={"n_estimators": [50, 100]})
            >>> print(df)
        """
        outer_cv = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        if param_grid is not None:
            inner_cv_splitter = StratifiedKFold(
                n_splits=inner_cv, shuffle=True, random_state=self.random_state
            )
            estimator_for_outer = GridSearchCV(
                estimator=self.estimator,
                param_grid=param_grid,
                cv=inner_cv_splitter,
                scoring=scoring,
                n_jobs=n_jobs,
                refit=True,
            )
        else:
            estimator_for_outer = self.estimator

        cv_results = cross_validate(
            estimator_for_outer,
            self.X,
            self.y,
            cv=outer_cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=n_jobs,
        )

        n_folds = len(cv_results["test_score"])
        df = pd.DataFrame(
            {
                "fold": list(range(1, n_folds + 1)),
                "train_score": cv_results["train_score"],
                "test_score": cv_results["test_score"],
            }
        )

        mean_test = df["test_score"].mean()
        std_test = df["test_score"].std()
        print(
            f"Nested CV ({scoring}): "
            f"{mean_test:.4f} +/- {std_test:.4f}"
        )
        return df

    def bootstrap_ci(
        self,
        metric: str = "balanced_accuracy",
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        fig_height: int = 400,
        fig_width: int = 700,
        font_size: int = 14,
    ) -> go.Figure:
        """Estimate performance confidence intervals via bootstrap resampling.

        The estimator is first fit on the full dataset. For each
        bootstrap iteration the out-of-bag (OOB) samples are used for
        evaluation. The resulting distribution and CI bounds are shown
        as a histogram.

        Supported ``metric`` values:
            - ``"accuracy"``
            - ``"balanced_accuracy"``
            - ``"f1_macro"``
            - ``"roc_auc_ovr"``

        Args:
            metric: Performance metric to compute on each bootstrap
                sample. Defaults to "balanced_accuracy".
            n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
            ci: Confidence level (e.g. 0.95 for 95% CI). Defaults to 0.95.
            fig_height: Height of the returned figure in pixels.
                Defaults to 400.
            fig_width: Width of the returned figure in pixels.
                Defaults to 700.
            font_size: Base font size. Defaults to 14.

        Returns:
            Plotly Figure showing the bootstrap score distribution with
            CI bounds annotated.

        Raises:
            ValueError: If ``metric`` is not one of the supported options.

        Examples:
            >>> fig = validator.bootstrap_ci(metric="f1_macro", n_bootstrap=500)
            >>> fig.show()
        """
        _supported = {"accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr"}
        if metric not in _supported:
            raise ValueError(
                f"metric must be one of {_supported}, got '{metric}'."
            )

        rng = np.random.default_rng(self.random_state)
        n_samples = len(self.y)

        self.estimator.fit(self.X, self.y)

        scores: list[float] = []
        for _ in range(n_bootstrap):
            in_bag = rng.integers(0, n_samples, size=n_samples)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[in_bag] = False

            # Fall back to all samples if OOB set is empty (very unlikely)
            if not oob_mask.any():
                oob_mask = ~oob_mask

            X_oob = self.X[oob_mask]
            y_oob = self.y[oob_mask]

            if metric == "roc_auc_ovr":
                if hasattr(self.estimator, "predict_proba"):
                    y_prob_oob = self.estimator.predict_proba(X_oob)
                    if self._n_classes == 2:
                        score = roc_auc_score(y_oob, y_prob_oob[:, 1])
                    else:
                        score = roc_auc_score(
                            y_oob,
                            y_prob_oob,
                            multi_class="ovr",
                            average="macro",
                        )
                else:
                    score = float("nan")
            else:
                y_pred_oob = self.estimator.predict(X_oob)
                if metric == "accuracy":
                    score = accuracy_score(y_oob, y_pred_oob)
                elif metric == "balanced_accuracy":
                    score = balanced_accuracy_score(y_oob, y_pred_oob)
                else:  # f1_macro
                    score = f1_score(
                        y_oob, y_pred_oob, average="macro", zero_division=0
                    )

            scores.append(score)

        scores_arr = np.array(scores, dtype=float)
        valid_scores = scores_arr[~np.isnan(scores_arr)]

        alpha = (1.0 - ci) / 2.0
        lower = float(np.percentile(valid_scores, alpha * 100))
        upper = float(np.percentile(valid_scores, (1.0 - alpha) * 100))

        self.bootstrap_scores_ = scores_arr
        self.bootstrap_ci_ = (lower, upper)

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=valid_scores,
                nbinsx=50,
                marker_color="steelblue",
                opacity=0.75,
                name="Bootstrap scores",
            )
        )
        fig.add_vline(
            x=lower,
            line_dash="dash",
            line_color="firebrick",
            annotation_text=f"Lower {ci:.0%} CI = {lower:.3f}",
            annotation_position="top left",
        )
        fig.add_vline(
            x=upper,
            line_dash="dash",
            line_color="firebrick",
            annotation_text=f"Upper {ci:.0%} CI = {upper:.3f}",
            annotation_position="top right",
        )
        fig.update_layout(
            title=(
                f"Bootstrap Distribution ({n_bootstrap} iterations) - {metric}<br>"
                f"{ci:.0%} CI: [{lower:.3f}, {upper:.3f}]"
            ),
            xaxis_title=metric,
            yaxis_title="Count",
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            showlegend=False,
        )
        return fig

    def confusion_matrix(
        self,
        normalize: bool = True,
        fig_height: int = 600,
        fig_width: int = 700,
        font_size: int = 14,
    ) -> go.Figure:
        """Compute and plot a confusion matrix from CV predictions.

        Predictions are obtained via ``cross_val_predict`` with a
        stratified k-fold splitter. When ``normalize=True`` each row is
        divided by its sum so that cells show proportions.

        Args:
            normalize: Whether to normalize rows to sum to 1.
                Defaults to True.
            fig_height: Height of the returned figure in pixels.
                Defaults to 600.
            fig_width: Width of the returned figure in pixels.
                Defaults to 700.
            font_size: Base font size. Defaults to 14.

        Returns:
            Plotly Figure containing an annotated heatmap of the
            confusion matrix.

        Examples:
            >>> fig = validator.confusion_matrix(normalize=False)
            >>> fig.show()
        """
        cv_splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        y_pred = cross_val_predict(
            self.estimator, self.X, self.y, cv=cv_splitter
        )

        cm = sk_confusion_matrix(self.y, y_pred, labels=self._unique_classes)

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm_display = cm.astype(float) / row_sums
            fmt = ".2f"
            colorbar_title = "Proportion"
        else:
            cm_display = cm.astype(float)
            fmt = ".0f"
            colorbar_title = "Count"

        annotations: list[dict] = []
        for i in range(cm_display.shape[0]):
            for j in range(cm_display.shape[1]):
                val = cm_display[i, j]
                text = f"{val:{fmt}}"
                annotations.append(
                    dict(
                        x=self.class_names[j],
                        y=self.class_names[i],
                        text=text,
                        showarrow=False,
                        font=dict(
                            color="white" if val > cm_display.max() / 2 else "black",
                            size=font_size,
                        ),
                    )
                )

        fig = go.Figure(
            data=go.Heatmap(
                z=cm_display,
                x=self.class_names,
                y=self.class_names,
                colorscale="Blues",
                colorbar=dict(title=colorbar_title),
            )
        )
        fig.update_layout(
            title=(
                "Confusion Matrix (Stratified {}-Fold CV{})"
                .format(self.cv, ", Normalized" if normalize else "")
            ),
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            annotations=annotations,
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
        )
        return fig

    def get_cv_summary(self) -> pd.DataFrame:
        """Compute cross-validation summary statistics.

        Evaluates the estimator using stratified k-fold CV and returns
        mean and standard deviation for accuracy, balanced accuracy,
        and macro F1.

        Returns:
            DataFrame with columns ``metric``, ``mean``, and ``std``,
            one row per metric.

        Examples:
            >>> summary = validator.get_cv_summary()
            >>> print(summary)
               metric      mean       std
            0  accuracy    0.920     0.031
            1  balanced_accuracy  0.918  0.033
            2  f1_macro    0.919     0.032
        """
        cv_splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        metrics = {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
        }

        rows: list[dict] = []
        for metric_name, scoring_key in metrics.items():
            fold_scores = cross_val_score(
                self.estimator,
                self.X,
                self.y,
                cv=cv_splitter,
                scoring=scoring_key,
            )
            rows.append(
                {
                    "metric": metric_name,
                    "mean": float(fold_scores.mean()),
                    "std": float(fold_scores.std()),
                }
            )

        return pd.DataFrame(rows)
