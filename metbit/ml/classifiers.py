# -*- coding: utf-8 -*-

__author__ = "aeiwz"
__copyright__ = "Copyright 2024, Theerayut"
__license__ = "MIT"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Development"


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance


_MODEL_REGISTRY = {
    "rf": lambda kw, rs: RandomForestClassifier(random_state=rs, **kw),
    "svm": lambda kw, rs: SVC(probability=True, random_state=rs, **kw),
    "xgb": lambda kw, rs: _make_xgb(kw, rs),
    "elasticnet": lambda kw, rs: LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        max_iter=1000,
        random_state=rs,
        **kw,
    ),
}


def _make_xgb(kw, rs):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for model='xgb'. Install it with: pip install xgboost"
        ) from exc
    return XGBClassifier(random_state=rs, eval_metric="mlogloss", **kw)


class MLClassifier:
    """Unified sklearn-compatible classifier for NMR metabolomics data.

    Wraps four model families (Random Forest, SVM, XGBoost, Elastic Net)
    behind a single interface with built-in cross-validation, feature
    importance, and Plotly visualisations.

    Attributes:
        pipeline_ (Pipeline): Fitted sklearn Pipeline (scaler + model).
        cv_results_ (dict): Cross-validated metrics stored after fit().
        classes_ (np.ndarray): Unique class labels observed during fit().

    Examples:
        >>> import pandas as pd
        >>> from metbit.ml import MLClassifier
        >>> X = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [4, 3, 2, 1]})
        >>> y = ["A", "A", "B", "B"]
        >>> clf = MLClassifier(X, y, model="rf")
        >>> clf.fit(cv=2)
        >>> preds = clf.predict(X)
    """

    def __init__(
        self,
        X: "pd.DataFrame | np.ndarray",
        y: "pd.Series | np.ndarray | list",
        model: str = "rf",
        features_name: "list | None" = None,
        scaling_method: str = "pareto",
        random_state: int = 42,
        **model_kwargs,
    ) -> None:
        """Initialise MLClassifier.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Target labels with length n_samples.
            model: Model family to use. One of ``"rf"``, ``"svm"``,
                ``"xgb"``, or ``"elasticnet"``. Defaults to ``"rf"``.
            features_name: Optional list of feature names. When *X* is a
                DataFrame the column names are used automatically.
            scaling_method: Scaling strategy. Currently StandardScaler is
                applied regardless of the value (kept for API compatibility
                with the rest of metbit). Defaults to ``"pareto"``.
            random_state: Random seed passed to the underlying estimator.
                Defaults to ``42``.
            **model_kwargs: Additional keyword arguments forwarded verbatim
                to the underlying sklearn/XGBoost estimator.

        Raises:
            ValueError: If *model* is not one of the supported families.

        Examples:
            >>> clf = MLClassifier(X, y, model="svm", C=1.0)
        """
        if model not in _MODEL_REGISTRY:
            raise ValueError(
                f"model must be one of {list(_MODEL_REGISTRY.keys())}, got '{model}'."
            )

        if isinstance(X, pd.DataFrame):
            self._feature_names: list = list(X.columns)
            self._X: np.ndarray = X.values
        else:
            self._X = np.asarray(X, dtype=float)
            self._feature_names = (
                features_name
                if features_name is not None
                else [f"feature_{i}" for i in range(self._X.shape[1])]
            )

        if features_name is not None:
            self._feature_names = list(features_name)

        from sklearn.preprocessing import LabelEncoder
        self._le = LabelEncoder()
        self._y_raw: np.ndarray = np.asarray(y)
        # XGBoost requires integer labels; encode always and decode predictions
        self._y: np.ndarray = self._le.fit_transform(self._y_raw)
        self._model_name: str = model
        self._scaling_method: str = scaling_method
        self._random_state: int = random_state
        self._model_kwargs: dict = model_kwargs

        estimator = _MODEL_REGISTRY[model](model_kwargs, random_state)
        self.pipeline_: Pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", estimator)]
        )

        self.cv_results_: dict = {}
        self.classes_: np.ndarray = self._le.classes_
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Core fit / predict
    # ------------------------------------------------------------------

    def fit(self, cv: int = 5) -> "MLClassifier":
        """Fit the model on the full dataset and compute cross-validated metrics.

        Cross-validation uses ``StratifiedKFold`` with *cv* folds. The
        following metrics are recorded in ``cv_results_``:

        - ``accuracy_mean`` / ``accuracy_std``
        - ``balanced_accuracy_mean`` / ``balanced_accuracy_std``
        - ``roc_auc_mean`` / ``roc_auc_std``

        Args:
            cv: Number of cross-validation folds. Defaults to ``5``.

        Returns:
            self - to allow method chaining.

        Examples:
            >>> clf = MLClassifier(X, y, model="rf").fit(cv=5)
            >>> print(clf.cv_results_)
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self._random_state)

        acc = cross_val_score(
            self.pipeline_, self._X, self._y, cv=skf, scoring="accuracy"
        )
        bal_acc = cross_val_score(
            self.pipeline_, self._X, self._y, cv=skf, scoring="balanced_accuracy"
        )

        # roc_auc_ovr requires probability support - all wrapped models have it
        roc_scoring = (
            "roc_auc_ovr" if len(self.classes_) > 2 else "roc_auc"
        )
        roc_auc = cross_val_score(
            self.pipeline_, self._X, self._y, cv=skf, scoring=roc_scoring
        )

        self.cv_results_ = {
            "accuracy_mean": float(np.mean(acc)),
            "accuracy_std": float(np.std(acc)),
            "balanced_accuracy_mean": float(np.mean(bal_acc)),
            "balanced_accuracy_std": float(np.std(bal_acc)),
            "roc_auc_mean": float(np.mean(roc_auc)),
            "roc_auc_std": float(np.std(roc_auc)),
        }

        self.pipeline_.fit(self._X, self._y)
        self._is_fitted = True
        return self

    def predict(self, X_new: "pd.DataFrame | np.ndarray") -> np.ndarray:
        """Predict class labels for *X_new*.

        Args:
            X_new: Feature matrix with shape (n_samples, n_features).

        Returns:
            Predicted class labels as a 1-D array.

        Examples:
            >>> labels = clf.predict(X_test)
        """
        self._check_fitted()
        return self._le.inverse_transform(self.pipeline_.predict(self._to_array(X_new)))

    def predict_proba(self, X_new: "pd.DataFrame | np.ndarray") -> np.ndarray:
        """Predict class-membership probabilities for *X_new*.

        Args:
            X_new: Feature matrix with shape (n_samples, n_features).

        Returns:
            Probability matrix with shape (n_samples, n_classes).

        Examples:
            >>> proba = clf.predict_proba(X_test)
        """
        self._check_fitted()
        return self.pipeline_.predict_proba(self._to_array(X_new))

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """Return a DataFrame of the top-N most important features.

        The importance source depends on the model family:

        - **RF / XGB**: ``feature_importances_`` from the fitted estimator.
        - **ElasticNet**: ``coef_`` (mean absolute value across classes for
          multi-class problems).
        - **SVM (linear kernel)**: ``coef_``.
        - **SVM (rbf / other kernel)**: permutation importance on training
          data (slower but kernel-agnostic).

        Args:
            top_n: Maximum number of features to return. Defaults to ``30``.

        Returns:
            DataFrame with columns ``feature`` and ``importance``, sorted
            by ``importance`` descending.

        Examples:
            >>> df = clf.get_feature_importance(top_n=20)
            >>> print(df.head())
        """
        self._check_fitted()
        estimator = self.pipeline_.named_steps["model"]
        names = self._feature_names

        if self._model_name in ("rf", "xgb"):
            importances = estimator.feature_importances_

        elif self._model_name == "elasticnet":
            coef = estimator.coef_
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

        elif self._model_name == "svm":
            kernel = getattr(estimator, "kernel", "rbf")
            if kernel == "linear":
                coef = estimator.coef_
                importances = (
                    np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef.ravel())
                )
            else:
                # permutation importance on (scaled) training data
                X_scaled = self.pipeline_.named_steps["scaler"].transform(self._X)
                result = permutation_importance(
                    estimator,
                    X_scaled,
                    self._y,
                    n_repeats=10,
                    random_state=self._random_state,
                )
                importances = result.importances_mean
        else:
            importances = np.zeros(len(names))

        df = pd.DataFrame({"feature": names, "importance": importances})
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return df.head(top_n)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_feature_importance(
        self,
        top_n: int = 30,
        fig_height: int = 700,
        fig_width: int = 900,
        font_size: int = 14,
    ) -> go.Figure:
        """Plot a horizontal bar chart of the top feature importances.

        Args:
            top_n: Number of top features to display. Defaults to ``30``.
            fig_height: Figure height in pixels. Defaults to ``700``.
            fig_width: Figure width in pixels. Defaults to ``900``.
            font_size: Base font size for axis labels and tick text.
                Defaults to ``14``.

        Returns:
            Plotly Figure object.

        Examples:
            >>> fig = clf.plot_feature_importance(top_n=20)
            >>> fig.show()
        """
        df = self.get_feature_importance(top_n=top_n)
        df_sorted = df.sort_values("importance", ascending=True)

        fig = go.Figure(
            go.Bar(
                x=df_sorted["importance"],
                y=df_sorted["feature"],
                orientation="h",
                marker=dict(
                    color=df_sorted["importance"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Importance"),
                ),
            )
        )
        fig.update_layout(
            title=dict(
                text=f"Feature Importance - Top {top_n} ({self._model_name.upper()})",
                font=dict(size=font_size + 2),
            ),
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            template="plotly_white",
            margin=dict(l=200, r=40, t=60, b=60),
        )
        return fig

    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        fig_height: int = 600,
        fig_width: int = 700,
        font_size: int = 14,
    ) -> go.Figure:
        """Plot a heatmap confusion matrix using training-set predictions.

        Args:
            normalize: Whether to normalize each row to sum to 1.0.
                Defaults to ``True``.
            fig_height: Figure height in pixels. Defaults to ``600``.
            fig_width: Figure width in pixels. Defaults to ``700``.
            font_size: Base font size. Defaults to ``14``.

        Returns:
            Plotly Figure object.

        Examples:
            >>> fig = clf.plot_confusion_matrix(normalize=True)
            >>> fig.show()
        """
        self._check_fitted()
        y_pred_enc = self.pipeline_.predict(self._X)
        int_labels = np.arange(len(self.classes_))
        cm = confusion_matrix(self._y, y_pred_enc, labels=int_labels)

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_plot = np.where(row_sums == 0, 0.0, cm / row_sums.astype(float))
            fmt_text = [[f"{v:.2f}" for v in row] for row in cm_plot]
            colorbar_title = "Proportion"
        else:
            cm_plot = cm.astype(float)
            fmt_text = [[str(int(v)) for v in row] for row in cm]
            colorbar_title = "Count"

        labels = [str(c) for c in self.classes_]

        fig = go.Figure(
            go.Heatmap(
                z=cm_plot,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=fmt_text,
                texttemplate="%{text}",
                textfont=dict(size=font_size),
                showscale=True,
                colorbar=dict(title=colorbar_title),
            )
        )
        fig.update_layout(
            title=dict(
                text="Confusion Matrix",
                font=dict(size=font_size + 2),
            ),
            xaxis=dict(title="Predicted label", tickfont=dict(size=font_size)),
            yaxis=dict(
                title="True label",
                tickfont=dict(size=font_size),
                autorange="reversed",
            ),
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            template="plotly_white",
        )
        return fig

    def plot_roc(
        self,
        fig_height: int = 600,
        fig_width: int = 800,
        font_size: int = 14,
    ) -> go.Figure:
        """Plot one-vs-rest ROC curves using cross-validated probability estimates.

        Each class gets its own curve with AUC displayed in the legend.
        A micro-average ROC curve is also included.

        Args:
            fig_height: Figure height in pixels. Defaults to ``600``.
            fig_width: Figure width in pixels. Defaults to ``800``.
            font_size: Base font size. Defaults to ``14``.

        Returns:
            Plotly Figure object.

        Examples:
            >>> fig = clf.plot_roc()
            >>> fig.show()
        """
        self._check_fitted()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self._random_state)
        y_score = cross_val_predict(
            self.pipeline_,
            self._X,
            self._y,
            cv=skf,
            method="predict_proba",
        )

        classes = self.classes_
        n_classes = len(classes)
        y_bin = label_binarize(self._y, classes=classes)
        if n_classes == 2:
            y_bin = np.hstack([1 - y_bin, y_bin])

        fig = go.Figure()
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        ]

        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc_val = auc(fpr, tpr)
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"Class {cls} (AUC = {roc_auc_val:.3f})",
                    line=dict(color=color, width=2),
                )
            )

        # micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        fig.add_trace(
            go.Scatter(
                x=fpr_micro,
                y=tpr_micro,
                mode="lines",
                name=f"Micro-average (AUC = {roc_auc_micro:.3f})",
                line=dict(color="black", width=2, dash="dash"),
            )
        )

        # random-chance diagonal
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random chance",
                line=dict(color="grey", width=1, dash="dot"),
                showlegend=True,
            )
        )

        fig.update_layout(
            title=dict(
                text="ROC Curves (One-vs-Rest, Cross-Validated)",
                font=dict(size=font_size + 2),
            ),
            xaxis=dict(
                title="False Positive Rate",
                range=[0, 1],
                tickfont=dict(size=font_size),
            ),
            yaxis=dict(
                title="True Positive Rate",
                range=[0, 1],
                tickfont=dict(size=font_size),
            ),
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            template="plotly_white",
            legend=dict(font=dict(size=font_size - 1)),
        )
        return fig

    # ------------------------------------------------------------------
    # Results accessor
    # ------------------------------------------------------------------

    def get_cv_results(self) -> dict:
        """Return the stored cross-validation metrics.

        Returns:
            Dictionary with keys ``accuracy_mean``, ``accuracy_std``,
            ``balanced_accuracy_mean``, ``balanced_accuracy_std``,
            ``roc_auc_mean``, and ``roc_auc_std``.

        Raises:
            RuntimeError: If ``fit()`` has not been called yet.

        Examples:
            >>> clf.fit()
            >>> print(clf.get_cv_results())
        """
        self._check_fitted()
        return self.cv_results_

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_array(X: "pd.DataFrame | np.ndarray") -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X, dtype=float)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before using this method.")
