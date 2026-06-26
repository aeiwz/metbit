"""Tests for the new stats, multivariate, ML, DL, and validation modules."""
# ruff: noqa: E501
from importlib import metadata as importlib_metadata
import platform
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


# ── Optional dependency availability (evaluated at collection time) ───────────

# torch
try:
    import torch as _torch_check  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_skip_no_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")

# xgboost — instantiate XGBClassifier to force libxgboost.dylib load.
# This catches the macOS libomp.dylib missing error at collection time
# rather than letting the test crash at runtime.
_XGB_SKIP_REASON: str = ""
if sys.version_info >= (3, 14):
    _XGB_AVAILABLE = False
    _XGB_SKIP_REASON = "xgboost segfaults under CPython 3.14 (upstream C-API incompatibility)"
elif platform.system() == "Darwin":
    _XGB_AVAILABLE = False
    _XGB_SKIP_REASON = "xgboost segfaults in _meta_from_numpy during sklearn CV on macOS arm64 (all versions)"
else:
    try:
        _xgb_version = importlib_metadata.version("xgboost")
        from xgboost import XGBClassifier as _XGBCheck
        _XGBCheck()
        _XGB_AVAILABLE = True
    except Exception as _e:
        _XGB_AVAILABLE = False
        _XGB_SKIP_REASON = f"xgboost not available: {_e}"

_skip_no_xgb = pytest.mark.skipif(not _XGB_AVAILABLE, reason=_XGB_SKIP_REASON)


# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def two_group_df():
    rng = np.random.default_rng(42)
    n = 40
    features = [f"f{i}" for i in range(20)]
    X = pd.DataFrame(rng.standard_normal((n, 20)), columns=features)
    X.iloc[:20] += 0.8  # make groups separable
    groups = ["A"] * 20 + ["B"] * 20
    df = X.copy()
    df["group"] = groups
    return df


@pytest.fixture
def three_group_df():
    rng = np.random.default_rng(42)
    n_per = 20
    features = [f"f{i}" for i in range(10)]
    dfs = []
    for i, g in enumerate(["A", "B", "C"]):
        block = pd.DataFrame(rng.standard_normal((n_per, 10)) + i, columns=features)
        block["group"] = g
        dfs.append(block)
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def X_y_binary():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((60, 30)), columns=[f"f{i}" for i in range(30)])
    X.iloc[:30] += 1.0
    y = pd.Series(["A"] * 30 + ["B"] * 30)
    return X, y


@pytest.fixture
def X_y_multi():
    X_arr, y_arr = make_classification(
        n_samples=90, n_features=20, n_informative=10,
        n_classes=3, n_clusters_per_class=1, random_state=0
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(20)])
    y = pd.Series([f"cls{c}" for c in y_arr])
    return X, y


# ── stats/multitest ──────────────────────────────────────────────────────────

class TestVolcanoPlot:
    def test_init_and_get_table(self, two_group_df):
        from metbit.stats.multitest import VolcanoPlot
        vp = VolcanoPlot(two_group_df, group_col="group")
        tbl = vp.get_table()
        assert isinstance(tbl, pd.DataFrame)
        assert "log2FC" in tbl.columns
        assert "p_value" in tbl.columns
        assert len(tbl) == 20  # 20 features

    def test_labels(self, two_group_df):
        from metbit.stats.multitest import VolcanoPlot
        vp = VolcanoPlot(two_group_df, group_col="group", fc_threshold=0.0,
                         p_value_threshold=0.99)
        tbl = vp.get_table()
        assert set(tbl["label"].unique()).issubset({"Up", "Down", "NS"})

    def test_plot_returns_figure(self, two_group_df):
        import plotly.graph_objects as go
        from metbit.stats.multitest import VolcanoPlot
        vp = VolcanoPlot(two_group_df, group_col="group")
        fig = vp.plot()
        assert isinstance(fig, go.Figure)

    def test_no_correction(self, two_group_df):
        from metbit.stats.multitest import VolcanoPlot
        vp = VolcanoPlot(two_group_df, group_col="group", correct_p=None)
        tbl = vp.get_table()
        assert "p_adj" in tbl.columns or "neg_log10_p" in tbl.columns

    def test_explicit_groups(self, two_group_df):
        from metbit.stats.multitest import VolcanoPlot
        vp = VolcanoPlot(two_group_df, group_col="group", group_a="A", group_b="B")
        tbl = vp.get_table()
        assert len(tbl) == 20


class TestANOVAStats:
    def test_fit_and_tables(self, three_group_df):
        from metbit.stats.multitest import ANOVAStats
        # melt to tidy format
        df_melt = three_group_df.melt(id_vars="group", var_name="feature", value_name="value")
        stat = ANOVAStats(df_melt, x_col="group", y_col="value").fit()
        tbl = stat.get_anova_table()
        assert "F" in tbl.columns or "f_stat" in tbl.columns or len(tbl) >= 1
        posthoc = stat.get_posthoc_table()
        assert isinstance(posthoc, pd.DataFrame)

    def test_plot(self, three_group_df):
        import plotly.graph_objects as go
        from metbit.stats.multitest import ANOVAStats
        df_melt = three_group_df.melt(id_vars="group", var_name="feature", value_name="value")
        fig = ANOVAStats(df_melt, x_col="group", y_col="value").fit().plot()
        assert isinstance(fig, go.Figure)

    def test_violin_plot(self, three_group_df):
        import plotly.graph_objects as go
        from metbit.stats.multitest import ANOVAStats
        df_melt = three_group_df.melt(id_vars="group", var_name="feature", value_name="value")
        fig = ANOVAStats(df_melt, x_col="group", y_col="value").fit().plot(plot_type="violin")
        assert isinstance(fig, go.Figure)


class TestKruskalStats:
    def test_fit_and_tables(self, three_group_df):
        from metbit.stats.multitest import KruskalStats
        df_melt = three_group_df.melt(id_vars="group", var_name="feature", value_name="value")
        stat = KruskalStats(df_melt, x_col="group", y_col="value").fit()
        tbl = stat.get_kruskal_table()
        assert isinstance(tbl, pd.DataFrame)
        posthoc = stat.get_posthoc_table()
        assert isinstance(posthoc, pd.DataFrame)

    def test_plot(self, three_group_df):
        import plotly.graph_objects as go
        from metbit.stats.multitest import KruskalStats
        df_melt = three_group_df.melt(id_vars="group", var_name="feature", value_name="value")
        fig = KruskalStats(df_melt, x_col="group", y_col="value").fit().plot()
        assert isinstance(fig, go.Figure)


# ── analysis/multivariate ────────────────────────────────────────────────────

class TestLDA:
    def test_fit_and_scores(self, X_y_binary):
        from metbit.analysis.multivariate import lda
        X, y = X_y_binary
        model = lda(X, y, n_components=1)
        model.fit()
        scores = model.get_scores()
        assert isinstance(scores, pd.DataFrame)
        assert len(scores) == len(X)

    def test_loadings(self, X_y_binary):
        from metbit.analysis.multivariate import lda
        X, y = X_y_binary
        model = lda(X, y)
        model.fit()
        loadings = model.get_loadings()
        assert isinstance(loadings, pd.DataFrame)

    def test_explained_variance(self, X_y_binary):
        from metbit.analysis.multivariate import lda
        X, y = X_y_binary
        model = lda(X, y)
        model.fit()
        ev = model.get_explained_variance()
        assert isinstance(ev, pd.DataFrame)

    def test_plot_scores(self, X_y_multi):
        import plotly.graph_objects as go
        from metbit.analysis.multivariate import lda
        X, y = X_y_multi
        model = lda(X, y)
        model.fit()
        fig = model.plot_lda_scores()
        assert isinstance(fig, go.Figure)

    def test_plot_loading(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.analysis.multivariate import lda
        X, y = X_y_binary
        model = lda(X, y)
        model.fit()
        fig = model.plot_loading_(ld=["LD1"])
        assert isinstance(fig, go.Figure)


class TestPLSR:
    def test_fit_and_predict(self):
        from metbit.analysis.multivariate import plsr
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((50, 20)))
        y = pd.Series(rng.standard_normal(50))
        model = plsr(X, y, n_components=2)
        model.fit()
        pred = model.predict(X)
        assert len(pred) == 50

    def test_scores_loadings_weights(self):
        from metbit.analysis.multivariate import plsr
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((50, 20)))
        y = pd.Series(rng.standard_normal(50))
        model = plsr(X, y, n_components=2)
        model.fit()
        assert isinstance(model.get_scores(), pd.DataFrame)
        assert isinstance(model.get_loadings(), pd.DataFrame)
        assert isinstance(model.get_weights(), pd.DataFrame)

    def test_metrics(self):
        from metbit.analysis.multivariate import plsr
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((50, 20)))
        y = pd.Series(rng.standard_normal(50))
        model = plsr(X, y, n_components=2)
        model.fit()
        m = model.get_metrics()
        assert "R2" in m

    def test_plots(self):
        import plotly.graph_objects as go
        from metbit.analysis.multivariate import plsr
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((50, 20)))
        y = pd.Series(rng.standard_normal(50))
        model = plsr(X, y, n_components=2)
        model.fit()
        assert isinstance(model.plot_predicted_vs_actual(), go.Figure)
        assert isinstance(model.plot_scores(), go.Figure)


class TestICA:
    def test_fit_and_components(self, X_y_binary):
        from metbit.analysis.multivariate import ica
        X, _ = X_y_binary
        model = ica(X, n_components=2)
        model.fit()
        comp = model.get_components()
        assert isinstance(comp, pd.DataFrame)
        assert comp.shape == (len(X), 2)

    def test_mixing(self, X_y_binary):
        from metbit.analysis.multivariate import ica
        X, _ = X_y_binary
        model = ica(X, n_components=2)
        model.fit()
        mix = model.get_mixing()
        assert isinstance(mix, pd.DataFrame)

    def test_plot(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.analysis.multivariate import ica
        X, y = X_y_binary
        model = ica(X, n_components=2)
        model.fit()
        fig = model.plot_components(color_=y)
        assert isinstance(fig, go.Figure)

    def test_plot_mixing(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.analysis.multivariate import ica
        X, _ = X_y_binary
        model = ica(X, n_components=2)
        model.fit()
        fig = model.plot_mixing_()
        assert isinstance(fig, go.Figure)


class TestHCA:
    def test_fit_and_cluster_labels(self, X_y_binary):
        from metbit.analysis.multivariate import hca
        X, y = X_y_binary
        model = hca(X, label=y)
        model.fit()
        labels = model.get_cluster_labels(n_clusters=2)
        assert len(labels) == len(X)

    def test_dendrogram(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.analysis.multivariate import hca
        X, _ = X_y_binary
        model = hca(X)
        model.fit()
        fig = model.plot_dendrogram()
        assert isinstance(fig, go.Figure)

    def test_heatmap(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.analysis.multivariate import hca
        X, _ = X_y_binary
        model = hca(X)
        model.fit()
        fig = model.plot_heatmap(n_clusters=2)
        assert isinstance(fig, go.Figure)


# ── ml/classifiers ───────────────────────────────────────────────────────────

class TestMLClassifier:
    @pytest.fixture
    def fitted_rf(self, X_y_binary):
        from metbit.ml.classifiers import MLClassifier
        X, y = X_y_binary
        return MLClassifier(X, y, model="rf", random_state=0).fit(cv=3)

    def test_rf_fit(self, fitted_rf):
        cv = fitted_rf.get_cv_results()
        assert "accuracy" in cv or len(cv) > 0

    def test_rf_predict(self, fitted_rf, X_y_binary):
        X, _ = X_y_binary
        preds = fitted_rf.predict(X)
        assert len(preds) == len(X)

    def test_rf_predict_proba(self, fitted_rf, X_y_binary):
        X, _ = X_y_binary
        proba = fitted_rf.predict_proba(X)
        assert proba.shape[0] == len(X)

    def test_rf_feature_importance(self, fitted_rf):
        fi = fitted_rf.get_feature_importance(top_n=10)
        assert isinstance(fi, pd.DataFrame)
        assert len(fi) <= 10

    def test_rf_plot_feature_importance(self, fitted_rf):
        import plotly.graph_objects as go
        fig = fitted_rf.plot_feature_importance(top_n=10)
        assert isinstance(fig, go.Figure)

    def test_rf_plot_confusion_matrix(self, fitted_rf):
        import plotly.graph_objects as go
        fig = fitted_rf.plot_confusion_matrix()
        assert isinstance(fig, go.Figure)

    def test_rf_plot_roc(self, fitted_rf):
        import plotly.graph_objects as go
        fig = fitted_rf.plot_roc()
        assert isinstance(fig, go.Figure)

    def test_svm_fit(self, X_y_binary):
        from metbit.ml.classifiers import MLClassifier
        X, y = X_y_binary
        clf = MLClassifier(X, y, model="svm", random_state=0).fit(cv=3)
        assert len(clf.predict(X)) == len(X)

    @_skip_no_xgb
    def test_xgb_fit(self, X_y_binary):
        from metbit.ml.classifiers import MLClassifier
        X, y = X_y_binary
        clf = MLClassifier(X, y, model="xgb", random_state=0).fit(cv=3)
        assert len(clf.predict(X)) == len(X)

    def test_elasticnet_fit(self, X_y_binary):
        from metbit.ml.classifiers import MLClassifier
        X, y = X_y_binary
        clf = MLClassifier(X, y, model="elasticnet", random_state=0).fit(cv=3)
        assert len(clf.predict(X)) == len(X)


# ── dl/models ────────────────────────────────────────────────────────────────

@_skip_no_torch
class TestSpectralAutoencoder:
    def test_fit_encode(self, X_y_binary):
        from metbit.dl.models import SpectralAutoencoder
        X, y = X_y_binary
        ae = SpectralAutoencoder(X, latent_dim=4, hidden_dims=[32], epochs=3,
                                 batch_size=16, random_state=0)
        ae.fit(verbose=False)
        emb = ae.encode()
        assert emb.shape == (len(X), 4)

    def test_reconstruct(self, X_y_binary):
        from metbit.dl.models import SpectralAutoencoder
        X, _ = X_y_binary
        ae = SpectralAutoencoder(X, latent_dim=4, hidden_dims=[32], epochs=3,
                                 batch_size=16, random_state=0)
        ae.fit(verbose=False)
        rec = ae.reconstruct()
        assert rec.shape == X.shape

    def test_plot_embedding(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.dl.models import SpectralAutoencoder
        X, y = X_y_binary
        ae = SpectralAutoencoder(X, latent_dim=4, hidden_dims=[32], epochs=3,
                                 batch_size=16, random_state=0)
        ae.fit(verbose=False)
        fig = ae.plot_embedding(color_=y)
        assert isinstance(fig, go.Figure)

    def test_plot_loss(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.dl.models import SpectralAutoencoder
        X, _ = X_y_binary
        ae = SpectralAutoencoder(X, latent_dim=4, hidden_dims=[32], epochs=3,
                                 batch_size=16, random_state=0)
        ae.fit(verbose=False)
        fig = ae.plot_loss()
        assert isinstance(fig, go.Figure)


@_skip_no_torch
class TestSpectralMLP:
    def test_fit_predict(self, X_y_binary):
        from metbit.dl.models import SpectralMLP
        X, y = X_y_binary
        mlp = SpectralMLP(X, y, hidden_dims=[32], epochs=3,
                          batch_size=16, dropout=0.1, random_state=0)
        mlp.fit(verbose=False)
        preds = mlp.predict()
        assert len(preds) == len(X)

    def test_predict_proba(self, X_y_binary):
        from metbit.dl.models import SpectralMLP
        X, y = X_y_binary
        mlp = SpectralMLP(X, y, hidden_dims=[32], epochs=3,
                          batch_size=16, dropout=0.1, random_state=0)
        mlp.fit(verbose=False)
        proba = mlp.predict_proba()
        assert proba.shape[0] == len(X)

    def test_plot_loss(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.dl.models import SpectralMLP
        X, y = X_y_binary
        mlp = SpectralMLP(X, y, hidden_dims=[32], epochs=3,
                          batch_size=16, dropout=0.1, random_state=0)
        mlp.fit(verbose=False)
        fig = mlp.plot_loss()
        assert isinstance(fig, go.Figure)

    def test_plot_confusion_matrix(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.dl.models import SpectralMLP
        X, y = X_y_binary
        mlp = SpectralMLP(X, y, hidden_dims=[32], epochs=3,
                          batch_size=16, dropout=0.1, random_state=0)
        mlp.fit(verbose=False)
        fig = mlp.plot_confusion_matrix()
        assert isinstance(fig, go.Figure)

    def test_get_accuracy(self, X_y_binary):
        from metbit.dl.models import SpectralMLP
        X, y = X_y_binary
        mlp = SpectralMLP(X, y, hidden_dims=[32], epochs=3,
                          batch_size=16, dropout=0.1, random_state=0)
        mlp.fit(verbose=False)
        acc = mlp.get_accuracy()
        assert 0.0 <= acc <= 1.0


@_skip_no_torch
class TestSpectralCNN:
    def test_fit_predict(self, X_y_binary):
        from metbit.dl.models import SpectralCNN
        X, y = X_y_binary
        cnn = SpectralCNN(X, y, filters=[8, 16], kernel_size=3, epochs=3,
                          batch_size=16, dropout=0.1, random_state=0)
        cnn.fit(verbose=False)
        preds = cnn.predict()
        assert len(preds) == len(X)

    def test_plot_loss(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.dl.models import SpectralCNN
        X, y = X_y_binary
        cnn = SpectralCNN(X, y, filters=[8, 16], kernel_size=3, epochs=3,
                          batch_size=16, dropout=0.1, random_state=0)
        cnn.fit(verbose=False)
        fig = cnn.plot_loss()
        assert isinstance(fig, go.Figure)

    def test_plot_confusion_matrix(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.dl.models import SpectralCNN
        X, y = X_y_binary
        cnn = SpectralCNN(X, y, filters=[8, 16], kernel_size=3, epochs=3,
                          batch_size=16, dropout=0.1, random_state=0)
        cnn.fit(verbose=False)
        fig = cnn.plot_confusion_matrix()
        assert isinstance(fig, go.Figure)


# ── validation/metrics ───────────────────────────────────────────────────────

class TestModelValidator:
    @pytest.fixture
    def validator(self, X_y_binary):
        from metbit.validation.metrics import ModelValidator
        X, y = X_y_binary
        est = RandomForestClassifier(n_estimators=10, random_state=0)
        return ModelValidator(est, X, y, cv=3, random_state=0)

    def test_roc_auc(self, validator):
        import plotly.graph_objects as go
        fig = validator.roc_auc()
        assert isinstance(fig, go.Figure)

    def test_confusion_matrix(self, validator):
        import plotly.graph_objects as go
        fig = validator.confusion_matrix()
        assert isinstance(fig, go.Figure)

    def test_confusion_matrix_unnormalized(self, validator):
        import plotly.graph_objects as go
        fig = validator.confusion_matrix(normalize=False)
        assert isinstance(fig, go.Figure)

    def test_bootstrap_ci(self, validator):
        import plotly.graph_objects as go
        fig = validator.bootstrap_ci(n_bootstrap=50)
        assert isinstance(fig, go.Figure)
        assert hasattr(validator, "bootstrap_scores_")
        assert hasattr(validator, "bootstrap_ci_")

    def test_nested_cv(self, validator):
        result = validator.nested_cv()
        assert isinstance(result, pd.DataFrame)
        assert "test_score" in result.columns

    def test_cv_summary(self, validator):
        summary = validator.get_cv_summary()
        assert isinstance(summary, pd.DataFrame)

    def test_multiclass(self, X_y_multi):
        import plotly.graph_objects as go
        from metbit.validation.metrics import ModelValidator
        X, y = X_y_multi
        est = RandomForestClassifier(n_estimators=10, random_state=0)
        val = ModelValidator(est, X, y, cv=3)
        fig = val.roc_auc()
        assert isinstance(fig, go.Figure)
        fig2 = val.confusion_matrix()
        assert isinstance(fig2, go.Figure)

    def test_nested_cv_with_param_grid(self, validator):
        from sklearn.ensemble import RandomForestClassifier
        from metbit.validation.metrics import ModelValidator
        X, y = next(iter([validator])), None  # use validator fixture's data
        # Re-create with param_grid
        import numpy as np, pandas as pd
        rng = np.random.default_rng(0)
        X2 = pd.DataFrame(rng.standard_normal((60, 10)))
        y2 = pd.Series(["A"] * 30 + ["B"] * 30)
        est = RandomForestClassifier(n_estimators=5, random_state=0)
        val = ModelValidator(est, X2, y2, cv=3)
        result = val.nested_cv(inner_cv=2, param_grid={"n_estimators": [5, 10]})
        assert "test_score" in result.columns

    def test_bootstrap_metrics(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.validation.metrics import ModelValidator
        X, y = X_y_binary
        est = RandomForestClassifier(n_estimators=10, random_state=0)
        val = ModelValidator(est, X, y, cv=3)
        for metric in ("accuracy", "f1_macro", "roc_auc_ovr"):
            fig = val.bootstrap_ci(metric=metric, n_bootstrap=20)
            assert isinstance(fig, go.Figure)

    def test_bad_class_names_raises(self, X_y_binary):
        from metbit.validation.metrics import ModelValidator
        X, y = X_y_binary
        est = RandomForestClassifier(n_estimators=5, random_state=0)
        with pytest.raises(ValueError, match="class_names"):
            ModelValidator(est, X, y, cv=3, class_names=["only_one"])


class TestMultivariateEdgeCases:
    def test_lda_bad_input_raises(self):
        from metbit.analysis.multivariate import lda
        import numpy as np
        with pytest.raises((ValueError, TypeError)):
            lda("not_array", [1, 2]).fit()

    def test_hca_different_methods(self):
        import plotly.graph_objects as go
        from metbit.analysis.multivariate import hca
        import numpy as np, pandas as pd
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((30, 10)))
        for method in ("average", "complete"):
            model = hca(X, method=method)
            model.fit()
            assert isinstance(model.plot_dendrogram(), go.Figure)

    def test_plsr_ndarray_input(self):
        from metbit.analysis.multivariate import plsr
        import numpy as np
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 15))
        y = rng.standard_normal(40)
        model = plsr(X, y, n_components=2)
        model.fit()
        assert model.predict(X).shape == (40,)

    def test_ica_ndarray_input(self):
        from metbit.analysis.multivariate import ica
        import numpy as np
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 20))
        model = ica(X, n_components=3)
        model.fit()
        assert model.get_components().shape == (50, 3)


class TestMLClassifierEdgeCases:
    def test_svm_feature_importance_rbf(self):
        import plotly.graph_objects as go
        from metbit.ml.classifiers import MLClassifier
        import numpy as np, pandas as pd
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((60, 10)))
        y = pd.Series(["A"] * 30 + ["B"] * 30)
        clf = MLClassifier(X, y, model="svm", kernel="rbf", random_state=0).fit(cv=3)
        fig = clf.plot_feature_importance()
        assert isinstance(fig, go.Figure)

    def test_elasticnet_feature_importance(self):
        import plotly.graph_objects as go
        from metbit.ml.classifiers import MLClassifier
        import numpy as np, pandas as pd
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((60, 10)))
        y = pd.Series(["A"] * 30 + ["B"] * 30)
        clf = MLClassifier(X, y, model="elasticnet", random_state=0).fit(cv=3)
        fig = clf.plot_feature_importance()
        assert isinstance(fig, go.Figure)

    def test_confusion_matrix_unnormalized(self):
        import plotly.graph_objects as go
        from metbit.ml.classifiers import MLClassifier
        import numpy as np, pandas as pd
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((60, 10)))
        y = pd.Series(["A"] * 30 + ["B"] * 30)
        clf = MLClassifier(X, y, model="rf", random_state=0).fit(cv=3)
        fig = clf.plot_confusion_matrix(normalize=False)
        assert isinstance(fig, go.Figure)


class TestMultitestEdgeCases:
    def test_star_annotations(self):
        from metbit.stats.multitest import ANOVAStats
        import numpy as np, pandas as pd
        rng = np.random.default_rng(42)
        # create data with clear significance
        vals = np.concatenate([rng.normal(0, 1, 30), rng.normal(5, 1, 30), rng.normal(10, 1, 30)])
        df = pd.DataFrame({"group": ["A"]*30 + ["B"]*30 + ["C"]*30, "value": vals})
        stat = ANOVAStats(df, x_col="group", y_col="value").fit()
        posthoc = stat.get_posthoc_table()
        assert len(posthoc) > 0

    def test_volcano_bonferroni(self, two_group_df):
        from metbit.stats.multitest import VolcanoPlot
        vp = VolcanoPlot(two_group_df, group_col="group", correct_p="bonferroni")
        tbl = vp.get_table()
        assert len(tbl) == 20

    def test_kruskal_no_correction(self, three_group_df):
        from metbit.stats.multitest import KruskalStats
        df_melt = three_group_df.melt(id_vars="group", var_name="feature", value_name="value")
        stat = KruskalStats(df_melt, x_col="group", y_col="value", correct_p=None).fit()
        assert isinstance(stat.get_posthoc_table(), pd.DataFrame)


class TestMultivariateInputCoverage:
    """Cover ndarray inputs, features_name, and error branches."""

    def test_lda_ndarray_with_features_name(self):
        from metbit.analysis.multivariate import lda
        import numpy as np
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 8))
        y = np.array(["A"] * 20 + ["B"] * 20)
        names = [f"feat_{i}" for i in range(8)]
        model = lda(X, y, features_name=names)
        model.fit()
        loadings = model.get_loadings()
        assert isinstance(loadings, pd.DataFrame)

    def test_lda_wrong_features_name_raises(self):
        from metbit.analysis.multivariate import lda
        import numpy as np
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 8))
        y = np.array(["A"] * 20 + ["B"] * 20)
        with pytest.raises(ValueError):
            lda(X, y, features_name=["only_one"]).fit()

    def test_lda_mismatched_samples_raises(self):
        from metbit.analysis.multivariate import lda
        import numpy as np
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 8))
        y = np.array(["A"] * 20)  # wrong length
        with pytest.raises(ValueError):
            lda(X, y).fit()

    def test_lda_color_series(self, X_y_multi):
        import plotly.graph_objects as go
        from metbit.analysis.multivariate import lda
        X, y = X_y_multi
        model = lda(X, y)
        model.fit()
        fig = model.plot_lda_scores(color_=y)
        assert isinstance(fig, go.Figure)

    def test_ica_no_color(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.analysis.multivariate import ica
        X, _ = X_y_binary
        model = ica(X, n_components=2)
        model.fit()
        fig = model.plot_components()  # no color
        assert isinstance(fig, go.Figure)

    def test_plsr_ndarray_features_name(self):
        from metbit.analysis.multivariate import plsr
        import numpy as np
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 10))
        y = rng.standard_normal(50)
        names = [f"v{i}" for i in range(10)]
        model = plsr(X, y, n_components=2, features_name=names)
        model.fit()
        assert isinstance(model.get_loadings(), pd.DataFrame)

    def test_hca_ndarray_input(self):
        import plotly.graph_objects as go
        from metbit.analysis.multivariate import hca
        import numpy as np
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 8))
        model = hca(X)
        model.fit()
        fig = model.plot_heatmap(n_clusters=2)
        assert isinstance(fig, go.Figure)

    def test_hca_no_colormap_threshold(self):
        import plotly.graph_objects as go
        from metbit.analysis.multivariate import hca
        import numpy as np, pandas as pd
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((20, 5)))
        model = hca(X)
        model.fit()
        fig = model.plot_dendrogram(color_threshold=1.0)
        assert isinstance(fig, go.Figure)


class TestMultitestBranchCoverage:
    def test_anova_custom_colors(self, three_group_df):
        import plotly.graph_objects as go
        from metbit.stats.multitest import ANOVAStats
        df_melt = three_group_df.melt(id_vars="group", var_name="feature", value_name="value")
        colors = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}
        fig = ANOVAStats(df_melt, x_col="group", y_col="value").fit().plot(custom_colors=colors)
        assert isinstance(fig, go.Figure)

    def test_volcano_explicit_value_cols(self, two_group_df):
        from metbit.stats.multitest import VolcanoPlot
        cols = [c for c in two_group_df.columns if c != "group"][:5]
        vp = VolcanoPlot(two_group_df, group_col="group", value_cols=cols)
        tbl = vp.get_table()
        assert len(tbl) == 5

    def test_volcano_zero_mean_group(self):
        from metbit.stats.multitest import VolcanoPlot
        import pandas as pd, numpy as np
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((40, 5)), columns=[f"f{i}" for i in range(5)])
        X.iloc[:20, 0] = 0.0  # force zero mean in group A for f0
        X["group"] = ["A"] * 20 + ["B"] * 20
        vp = VolcanoPlot(X, group_col="group")
        tbl = vp.get_table()
        assert len(tbl) == 5

    def test_star_levels(self):
        from metbit.stats.multitest import _p_to_stars  # internal helper
        assert _p_to_stars(0.0001) == "***"
        assert _p_to_stars(0.005) == "**"
        assert _p_to_stars(0.03) == "*"
        assert _p_to_stars(0.5) == "ns"

    def test_volcano_bad_group_col_raises(self, two_group_df):
        from metbit.stats.multitest import VolcanoPlot
        with pytest.raises(ValueError):
            VolcanoPlot(two_group_df, group_col="nonexistent")

    def test_kruskal_custom_group_order(self, three_group_df):
        import plotly.graph_objects as go
        from metbit.stats.multitest import KruskalStats
        df_melt = three_group_df.melt(id_vars="group", var_name="feature", value_name="value")
        stat = KruskalStats(df_melt, x_col="group", y_col="value",
                            group_order=["C", "B", "A"]).fit()
        fig = stat.plot()
        assert isinstance(fig, go.Figure)


class TestMLClassifierEdgeCoverage:
    def test_svm_linear_feature_importance(self):
        import plotly.graph_objects as go
        from metbit.ml.classifiers import MLClassifier
        import numpy as np, pandas as pd
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((60, 10)))
        y = pd.Series(["A"] * 30 + ["B"] * 30)
        clf = MLClassifier(X, y, model="svm", kernel="linear", random_state=0).fit(cv=3)
        fi = clf.get_feature_importance()
        assert isinstance(fi, pd.DataFrame)

    def test_ndarray_input(self):
        from metbit.ml.classifiers import MLClassifier
        import numpy as np
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 10))
        y = np.array(["A"] * 30 + ["B"] * 30)
        clf = MLClassifier(X, y, model="rf", random_state=0).fit(cv=3)
        assert len(clf.predict(X)) == 60

    def test_bad_model_raises(self):
        from metbit.ml.classifiers import MLClassifier
        import numpy as np, pandas as pd
        X = pd.DataFrame(np.zeros((10, 5)))
        y = pd.Series(["A"] * 5 + ["B"] * 5)
        with pytest.raises(ValueError):
            MLClassifier(X, y, model="notamodel")

    def test_predict_before_fit_raises(self):
        from metbit.ml.classifiers import MLClassifier
        import numpy as np, pandas as pd
        X = pd.DataFrame(np.zeros((10, 5)))
        y = pd.Series(["A"] * 5 + ["B"] * 5)
        clf = MLClassifier(X, y, model="rf")
        with pytest.raises(RuntimeError):
            clf.predict(X)


class TestFinalCoverageGaps:
    """Micro-tests to close remaining branch gaps."""

    # multitest error/branch paths
    def test_anova_bad_plot_type_raises(self, three_group_df):
        from metbit.stats.multitest import ANOVAStats
        df = three_group_df.melt(id_vars="group", var_name="f", value_name="v")
        with pytest.raises(ValueError):
            ANOVAStats(df, x_col="group", y_col="v").fit().plot(plot_type="scatter")

    def test_anova_not_fitted_raises(self, three_group_df):
        from metbit.stats.multitest import ANOVAStats
        df = three_group_df.melt(id_vars="group", var_name="f", value_name="v")
        with pytest.raises(RuntimeError):
            ANOVAStats(df, x_col="group", y_col="v").get_anova_table()

    def test_kruskal_not_fitted_raises(self, three_group_df):
        from metbit.stats.multitest import KruskalStats
        df = three_group_df.melt(id_vars="group", var_name="f", value_name="v")
        with pytest.raises(RuntimeError):
            KruskalStats(df, x_col="group", y_col="v").get_kruskal_table()

    def test_kruskal_violin(self, three_group_df):
        import plotly.graph_objects as go
        from metbit.stats.multitest import KruskalStats
        df = three_group_df.melt(id_vars="group", var_name="f", value_name="v")
        fig = KruskalStats(df, x_col="group", y_col="v").fit().plot(plot_type="violin")
        assert isinstance(fig, go.Figure)

    def test_anova_repr(self, three_group_df):
        from metbit.stats.multitest import ANOVAStats
        df = three_group_df.melt(id_vars="group", var_name="f", value_name="v")
        s = repr(ANOVAStats(df, x_col="group", y_col="v"))
        assert "ANOVAStats" in s

    def test_kruskal_repr(self, three_group_df):
        from metbit.stats.multitest import KruskalStats
        df = three_group_df.melt(id_vars="group", var_name="f", value_name="v")
        s = repr(KruskalStats(df, x_col="group", y_col="v"))
        assert "KruskalStats" in s

    def test_volcano_too_many_groups_raises(self):
        from metbit.stats.multitest import VolcanoPlot
        import numpy as np, pandas as pd
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((30, 5)), columns=[f"f{i}" for i in range(5)])
        df["group"] = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
        with pytest.raises(ValueError):
            VolcanoPlot(df, group_col="group")

    # multivariate error paths
    def test_lda_bad_y_raises(self):
        from metbit.analysis.multivariate import lda
        import numpy as np
        X = np.random.default_rng(0).standard_normal((30, 5))
        with pytest.raises((ValueError, TypeError)):
            lda(X, "notarray").fit()

    def test_ica_no_features_name(self, X_y_binary):
        from metbit.analysis.multivariate import ica
        X, _ = X_y_binary
        model = ica(X.values, n_components=2)
        model.fit()
        assert model.get_components().shape[1] == 2

    def test_hca_bad_X_raises(self):
        from metbit.analysis.multivariate import hca
        with pytest.raises((ValueError, TypeError)):
            hca("notarray").fit()

    def test_plsr_bad_X_raises(self):
        from metbit.analysis.multivariate import plsr
        with pytest.raises((ValueError, TypeError)):
            plsr("notarray", [1, 2, 3]).fit()

    # ml edge paths
    def test_ml_features_name_override(self):
        from metbit.ml.classifiers import MLClassifier
        import numpy as np, pandas as pd
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 5))
        y = np.array(["A"] * 20 + ["B"] * 20)
        names = ["a", "b", "c", "d", "e"]
        clf = MLClassifier(X, y, model="rf", features_name=names, random_state=0).fit(cv=3)
        fi = clf.get_feature_importance()
        assert list(fi["feature"]) == sorted(names, key=lambda n: -fi.set_index("feature").loc[n, "importance"])

    # validation edge paths
    def test_validator_valid_class_names(self, X_y_binary):
        from metbit.validation.metrics import ModelValidator
        X, y = X_y_binary
        est = RandomForestClassifier(n_estimators=5, random_state=0)
        val = ModelValidator(est, X, y, cv=3, class_names=["A", "B"])
        assert val.class_names == ["A", "B"]

    def test_bootstrap_roc_auc_multiclass(self, X_y_multi):
        import plotly.graph_objects as go
        from metbit.validation.metrics import ModelValidator
        X, y = X_y_multi
        est = RandomForestClassifier(n_estimators=10, random_state=0)
        val = ModelValidator(est, X, y, cv=3)
        fig = val.bootstrap_ci(metric="roc_auc_ovr", n_bootstrap=20)
        assert isinstance(fig, go.Figure)

    def test_bootstrap_bad_metric_raises(self, X_y_binary):
        from metbit.validation.metrics import ModelValidator
        X, y = X_y_binary
        est = RandomForestClassifier(n_estimators=5, random_state=0)
        val = ModelValidator(est, X, y, cv=3)
        with pytest.raises(ValueError):
            val.bootstrap_ci(metric="invalid_metric")


@_skip_no_torch
class TestDLEdgeCases:
    def test_autoencoder_ndarray(self):
        from metbit.dl.models import SpectralAutoencoder
        import numpy as np
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 20)).astype(np.float32)
        ae = SpectralAutoencoder(X, latent_dim=4, hidden_dims=[16], epochs=2,
                                 batch_size=10, random_state=0)
        ae.fit(verbose=False)
        emb = ae.encode(X)
        assert emb.shape == (40, 4)

    def test_mlp_multiclass(self, X_y_multi):
        from metbit.dl.models import SpectralMLP
        X, y = X_y_multi
        mlp = SpectralMLP(X, y, hidden_dims=[32], epochs=2, batch_size=16,
                          dropout=0.0, random_state=0)
        mlp.fit(verbose=False)
        assert mlp.get_accuracy() >= 0.0

    def test_cnn_predict_proba(self, X_y_binary):
        from metbit.dl.models import SpectralCNN
        X, y = X_y_binary
        cnn = SpectralCNN(X, y, filters=[8], kernel_size=3, epochs=2,
                          batch_size=16, dropout=0.0, random_state=0)
        cnn.fit(verbose=False)
        proba = cnn.predict_proba()
        assert proba.shape[0] == len(X)

    def test_autoencoder_verbose(self, X_y_binary):
        from metbit.dl.models import SpectralAutoencoder
        X, _ = X_y_binary
        ae = SpectralAutoencoder(X, latent_dim=4, hidden_dims=[16], epochs=3,
                                 batch_size=16, random_state=0)
        ae.fit(verbose=True)  # covers print branch
        assert len(ae.training_loss_) == 3

    def test_mlp_verbose(self, X_y_binary):
        from metbit.dl.models import SpectralMLP
        X, y = X_y_binary
        mlp = SpectralMLP(X, y, hidden_dims=[16], epochs=3,
                          batch_size=16, dropout=0.0, random_state=0)
        mlp.fit(verbose=True)
        assert mlp.get_accuracy() >= 0.0

    def test_cnn_verbose(self, X_y_binary):
        from metbit.dl.models import SpectralCNN
        X, y = X_y_binary
        cnn = SpectralCNN(X, y, filters=[8], kernel_size=3, epochs=3,
                          batch_size=16, dropout=0.0, random_state=0)
        cnn.fit(verbose=True)
        assert cnn.get_accuracy() >= 0.0

    def test_autoencoder_encode_with_X(self, X_y_binary):
        from metbit.dl.models import SpectralAutoencoder
        X, _ = X_y_binary
        ae = SpectralAutoencoder(X, latent_dim=4, hidden_dims=[16], epochs=2,
                                 batch_size=16, random_state=0)
        ae.fit(verbose=False)
        import numpy as np
        X_new = X.values[:10]
        emb = ae.encode(X_new)  # covers _to_numpy path
        assert emb.shape == (10, 4)
        rec = ae.reconstruct(X_new)
        assert rec.shape[0] == 10

    def test_mlp_predict_with_X_arg(self, X_y_binary):
        from metbit.dl.models import SpectralMLP
        X, y = X_y_binary
        mlp = SpectralMLP(X, y, hidden_dims=[16], epochs=2,
                          batch_size=16, dropout=0.0, random_state=0)
        mlp.fit(verbose=False)
        preds = mlp.predict(X.values[:5])  # numpy array path
        assert len(preds) == 5

    def test_cnn_predict_with_X_arg(self, X_y_binary):
        from metbit.dl.models import SpectralCNN
        X, y = X_y_binary
        cnn = SpectralCNN(X, y, filters=[8], kernel_size=3, epochs=2,
                          batch_size=16, dropout=0.0, random_state=0)
        cnn.fit(verbose=False)
        preds = cnn.predict(X.values[:5])  # numpy array path
        assert len(preds) == 5

    def test_cnn_unnormalized_confusion(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.dl.models import SpectralCNN
        X, y = X_y_binary
        cnn = SpectralCNN(X, y, filters=[8], kernel_size=3, epochs=2,
                          batch_size=16, dropout=0.0, random_state=0)
        cnn.fit(verbose=False)
        fig = cnn.plot_confusion_matrix(normalize=False)
        assert isinstance(fig, go.Figure)

    def test_mlp_unnormalized_confusion(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.dl.models import SpectralMLP
        X, y = X_y_binary
        mlp = SpectralMLP(X, y, hidden_dims=[16], epochs=2,
                          batch_size=16, dropout=0.0, random_state=0)
        mlp.fit(verbose=False)
        fig = mlp.plot_confusion_matrix(normalize=False)
        assert isinstance(fig, go.Figure)

    def test_autoencoder_plot_embedding_no_color(self, X_y_binary):
        import plotly.graph_objects as go
        from metbit.dl.models import SpectralAutoencoder
        X, _ = X_y_binary
        ae = SpectralAutoencoder(X, latent_dim=6, hidden_dims=[16], epochs=2,
                                 batch_size=16, random_state=0)
        ae.fit(verbose=False)
        fig = ae.plot_embedding(components=[0, 2])
        assert isinstance(fig, go.Figure)


class TestMultivariateRemainingBranches:
    def test_lda_no_features_name_ndarray(self):
        from metbit.analysis.multivariate import lda
        import numpy as np
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 6))
        y = np.array(["A"] * 20 + ["B"] * 20)
        model = lda(X, y)  # no features_name → list(range(...))
        model.fit()
        assert model.get_scores().shape[0] == 40

    def test_lda_max_components_for_binary(self, X_y_binary):
        from metbit.analysis.multivariate import lda
        X, y = X_y_binary
        # binary class → exactly 1 component
        model = lda(X, y, n_components=1)
        model.fit()
        ev = model.get_explained_variance()
        # sklearn LDA clips to n_classes-1; fallback path sets evr to uniform
        assert isinstance(ev, pd.DataFrame)

    def test_lda_color_auto_palette(self, X_y_multi):
        import plotly.graph_objects as go
        from metbit.analysis.multivariate import lda
        X, y = X_y_multi
        model = lda(X, y)
        model.fit()
        # no color_dict → triggers auto-palette
        fig = model.plot_lda_scores(color_=y)
        assert isinstance(fig, go.Figure)

    def test_plsr_wrong_y_shape_raises(self):
        from metbit.analysis.multivariate import plsr
        import numpy as np, pandas as pd
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((50, 10)))
        y = rng.standard_normal(30)  # wrong length
        with pytest.raises((ValueError, Exception)):
            plsr(X, y).fit()

    def test_ica_wrong_X_raises(self):
        from metbit.analysis.multivariate import ica
        with pytest.raises((ValueError, TypeError)):
            ica("bad", n_components=2).fit()

    def test_hca_features_name_ndarray(self):
        from metbit.analysis.multivariate import hca
        import numpy as np
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 4))
        names = ["a", "b", "c", "d"]
        model = hca(X, features_name=names)
        model.fit()
        labels = model.get_cluster_labels(n_clusters=2)
        assert len(labels) == 20
