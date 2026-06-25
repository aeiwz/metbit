# -*- coding: utf-8 -*-

__author__ = 'aeiwz'

from typing import Optional, List, Union, Any
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LDA
from sklearn.cross_decomposition import PLSRegression as _PLSRegression
from sklearn.decomposition import FastICA as _FastICA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as plotly_colours
import warnings
warnings.filterwarnings('ignore')

from ..preprocessing.scaler_ext import Scaler
from .opls_da import _resolve_scale_power


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_color_dict(groups):
    """Build an automatic color mapping for a sequence of unique group labels."""
    name_color_set = [
        'Plotly', 'D3', 'G10', 'T10', 'Alphabet', 'Dark24', 'Light24',
        'Set1', 'Pastel1', 'Dark2', 'Set2', 'Pastel2', 'Set3',
        'Antique', 'Safe', 'Bold', 'Pastel', 'Vivid', 'Prism',
    ]
    palette = []
    for name in name_color_set:
        palette += getattr(plotly_colours.qualitative, name)
    unique = list(dict.fromkeys(groups))
    return {g: palette[i % len(palette)] for i, g in enumerate(unique)}


# ---------------------------------------------------------------------------
# CLASS 1: lda
# ---------------------------------------------------------------------------

class lda:
    """Linear Discriminant Analysis (LDA) for supervised dimensionality reduction.

    Parameters:
        X: Feature matrix (DataFrame or ndarray), rows=samples, cols=features.
        y: Class labels (Series, ndarray, or list).
        features_name: Optional feature names. Inferred from X columns when X is
            a DataFrame and features_name is None.
        n_components: Number of LD components to retain. Defaults to n_classes - 1.
        scaling_method: One of "pareto", "mean", "uv", "minmax", or None.
        random_state: Unused directly but kept for API consistency.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.analysis.multivariate import lda
        >>> X = pd.DataFrame(np.random.rand(60, 50))
        >>> y = pd.Series(['A'] * 20 + ['B'] * 20 + ['C'] * 20)
        >>> model = lda(X=X, y=y, n_components=2)
        >>> model.fit()
        >>> scores = model.get_scores()
        >>> fig = model.plot_lda_scores()
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List[Any]],
        features_name: Optional[Union[pd.Series, np.ndarray, List[Any]]] = None,
        n_components: Optional[int] = None,
        scaling_method: str = 'pareto',
        random_state: int = 42,
    ) -> None:
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError('X must be a DataFrame or ndarray')
        if not isinstance(y, (pd.Series, np.ndarray, list)):
            raise ValueError('y must be a Series, ndarray, or list')
        if len(y) != X.shape[0]:
            raise ValueError('X and y must have the same number of samples')

        if features_name is not None:
            if len(features_name) != X.shape[1]:
                raise ValueError('features_name length must match X.shape[1]')
            self.features_name = list(features_name)
        elif isinstance(X, pd.DataFrame):
            self.features_name = list(X.columns)
        else:
            self.features_name = list(range(X.shape[1]))

        self.X = X
        self.y = y if isinstance(y, pd.Series) else pd.Series(y)
        self.n_components = n_components
        self.scaling_method = scaling_method
        self.random_state = random_state

    def fit(self) -> None:
        """Fit the LDA model to the scaled data.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import lda
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> y = pd.Series(['A'] * 30 + ['B'] * 30)
            >>> model = lda(X=X, y=y)
            >>> model.fit()
        """
        X = self.X
        y = self.y
        features_name = self.features_name

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=features_name)

        scale_power = _resolve_scale_power(self.scaling_method)
        scaler = Scaler(scale_power=scale_power)
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        n_classes = y.nunique()
        n_components = self.n_components if self.n_components is not None else (n_classes - 1)

        lda_model = _LDA(n_components=n_components)
        lda_model.fit(X_scaled, y)

        scores = lda_model.transform(X_scaled)
        # coef_ shape: (n_classes, n_features) for multi-class
        # scalings_ shape: (n_features, n_components)
        loadings = lda_model.scalings_

        ld_cols = [f'LD{i+1}' for i in range(n_components)]

        df_scores = pd.DataFrame(scores, columns=ld_cols, index=y.index)
        df_scores['Group'] = y.values

        df_loadings = pd.DataFrame(loadings, index=features_name, columns=ld_cols)
        df_loadings['Features'] = features_name

        # Explained variance ratio
        if hasattr(lda_model, 'explained_variance_ratio_'):
            evr = lda_model.explained_variance_ratio_
        else:
            evr = np.ones(n_components) / n_components

        df_ev = pd.DataFrame({
            'LD': ld_cols,
            'Explained variance': evr,
            'Cumulative variance': np.cumsum(evr),
        })

        self.lda_model = lda_model
        self.scaler_ = scaler
        self.df_scores_ = df_scores
        self.df_loadings_ = df_loadings
        self.df_explained_variance_ = df_ev
        self.n_components_ = n_components

    def get_scores(self) -> pd.DataFrame:
        """Return the LD scores DataFrame.

        Returns:
            DataFrame of shape (n_samples, n_components + 1) with LD columns and a
            'Group' column.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import lda
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> y = pd.Series(['A'] * 30 + ['B'] * 30)
            >>> model = lda(X=X, y=y)
            >>> model.fit()
            >>> df = model.get_scores()
        """
        return self.df_scores_

    def get_loadings(self) -> pd.DataFrame:
        """Return the LD loadings (scalings) DataFrame.

        Returns:
            DataFrame of shape (n_features, n_components) indexed by feature names.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import lda
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> y = pd.Series(['A'] * 30 + ['B'] * 30)
            >>> model = lda(X=X, y=y)
            >>> model.fit()
            >>> df = model.get_loadings()
        """
        return self.df_loadings_

    def get_explained_variance(self) -> pd.DataFrame:
        """Return the explained variance ratio per LD component.

        Returns:
            DataFrame with columns 'LD', 'Explained variance', 'Cumulative variance'.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import lda
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> y = pd.Series(['A'] * 30 + ['B'] * 30)
            >>> model = lda(X=X, y=y)
            >>> model.fit()
            >>> df = model.get_explained_variance()
        """
        return self.df_explained_variance_

    def plot_lda_scores(
        self,
        ld: List[str] = ['LD1', 'LD2'],
        color_: Optional[pd.Series] = None,
        color_dict: Optional[dict] = None,
        marker_size: int = 35,
        fig_height: int = 900,
        fig_width: int = 1300,
        font_size: int = 20,
    ) -> go.Figure:
        """Plot LDA scores scatter.

        Parameters:
            ld: Two LD component names to plot on x and y axes.
            color_: Optional alternative grouping series for colouring points.
            color_dict: Optional mapping of group label to colour hex string.
            marker_size: Marker diameter in pixels.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Global font size.

        Returns:
            Plotly Figure object.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import lda
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> y = pd.Series(['A'] * 30 + ['B'] * 30)
            >>> model = lda(X=X, y=y)
            >>> model.fit()
            >>> fig = model.plot_lda_scores(ld=['LD1', 'LD2'])
            >>> fig.show()
        """
        df = self.df_scores_.copy()

        if color_ is not None:
            df['Group'] = color_.values if isinstance(color_, pd.Series) else color_

        if color_dict is None:
            color_dict = _build_color_dict(df['Group'].tolist())

        ev = self.df_explained_variance_.set_index('LD')

        def _ev_label(col):
            if col in ev.index:
                pct = round(ev.loc[col, 'Explained variance'] * 100, 2)
                return f'{col} ({pct}%)'
            return col

        fig = px.scatter(
            df,
            x=ld[0],
            y=ld[1],
            color='Group',
            color_discrete_map=color_dict,
            title=f'<b>LDA Scores Plot</b> ({self.scaling_method} scaling)',
            height=fig_height,
            width=fig_width,
            labels={ld[0]: _ev_label(ld[0]), ld[1]: _ev_label(ld[1])},
        )
        fig.update_traces(
            marker=dict(
                size=marker_size,
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey'),
            )
        )
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black',
                         showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black',
                         showline=True, linewidth=2, linecolor='black')
        fig.update_layout(
            title={'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            font=dict(size=font_size),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    def plot_loading_(
        self,
        ld: List[str] = ['LD1', 'LD2'],
        fig_height: int = 600,
        fig_width: int = 1800,
        font_size: int = 20,
    ) -> go.Figure:
        """Plot LDA loadings as a scatter over features.

        Parameters:
            ld: LD component names to overlay.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Global font size.

        Returns:
            Plotly Figure object.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import lda
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> y = pd.Series(['A'] * 30 + ['B'] * 30)
            >>> model = lda(X=X, y=y)
            >>> model.fit()
            >>> fig = model.plot_loading_(ld=['LD1', 'LD2'])
            >>> fig.show()
        """
        df = self.df_loadings_.copy()
        fig = px.scatter(
            df,
            x='Features',
            y=ld,
            height=fig_height,
            width=fig_width,
            title='LDA Loadings Plot',
        )
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
        fig.update_layout(
            title={'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            font=dict(size=font_size),
        )
        fig.update_traces(marker=dict(size=3))
        return fig


# ---------------------------------------------------------------------------
# CLASS 2: plsr
# ---------------------------------------------------------------------------

class plsr:
    """PLS Regression for continuous response prediction.

    Parameters:
        X: Feature matrix (DataFrame or ndarray), rows=samples, cols=features.
        y: Continuous response variable (numeric Series, ndarray, or list).
        features_name: Optional feature names.
        n_components: Number of latent components. Default is 2.
        scaling_method: One of "pareto", "mean", "uv", "minmax", or None.
        random_state: Unused directly but kept for API consistency.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.analysis.multivariate import plsr
        >>> X = pd.DataFrame(np.random.rand(60, 50))
        >>> y = pd.Series(np.random.rand(60))
        >>> model = plsr(X=X, y=y, n_components=2)
        >>> model.fit()
        >>> metrics = model.get_metrics()
        >>> fig = model.plot_predicted_vs_actual()
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List[Any]],
        features_name: Optional[Union[pd.Series, np.ndarray, List[Any]]] = None,
        n_components: int = 2,
        scaling_method: str = 'pareto',
        random_state: int = 42,
    ) -> None:
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError('X must be a DataFrame or ndarray')
        if not isinstance(y, (pd.Series, np.ndarray, list)):
            raise ValueError('y must be a Series, ndarray, or list')
        if len(y) != X.shape[0]:
            raise ValueError('X and y must have the same number of samples')

        if features_name is not None:
            if len(features_name) != X.shape[1]:
                raise ValueError('features_name length must match X.shape[1]')
            self.features_name = list(features_name)
        elif isinstance(X, pd.DataFrame):
            self.features_name = list(X.columns)
        else:
            self.features_name = list(range(X.shape[1]))

        self.X = X
        self.y = np.array(y, dtype=float)
        self.n_components = n_components
        self.scaling_method = scaling_method
        self.random_state = random_state

    def fit(self) -> None:
        """Fit the PLS Regression model and compute cross-validated Q2.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import plsr
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> y = pd.Series(np.random.rand(60))
            >>> model = plsr(X=X, y=y, n_components=2)
            >>> model.fit()
        """
        X = self.X
        y = self.y
        features_name = self.features_name

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=features_name)

        scale_power = _resolve_scale_power(self.scaling_method)
        scaler = Scaler(scale_power=scale_power)
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        pls = _PLSRegression(n_components=self.n_components)
        pls.fit(X_scaled, y)

        y_pred = pls.predict(X_scaled).ravel()
        r2 = r2_score(y, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

        # LeaveOneOut Q2
        loo = LeaveOneOut()
        y_loo = np.zeros_like(y)
        for train_idx, test_idx in loo.split(X_scaled):
            pls_loo = _PLSRegression(n_components=self.n_components)
            pls_loo.fit(X_scaled[train_idx], y[train_idx])
            y_loo[test_idx] = pls_loo.predict(X_scaled[test_idx]).ravel()
        q2 = float(r2_score(y, y_loo))

        # T scores (X scores)
        t_scores = pls.x_scores_
        # P loadings (X loadings)
        p_loadings = pls.x_loadings_
        # W weights (X weights)
        w_weights = pls.x_weights_

        lv_cols = [f'LV{i+1}' for i in range(self.n_components)]

        df_scores = pd.DataFrame(t_scores, columns=lv_cols)
        df_loadings = pd.DataFrame(p_loadings, index=features_name, columns=lv_cols)
        df_weights = pd.DataFrame(w_weights, index=features_name, columns=lv_cols)

        self.pls_model = pls
        self.scaler_ = scaler
        self.X_scaled_ = X_scaled
        self.y_pred_ = y_pred
        self.df_scores_ = df_scores
        self.df_loadings_ = df_loadings
        self.df_weights_ = df_weights
        self.metrics_ = {'R2': r2, 'Q2': q2, 'RMSE': rmse}

    def predict(self, X_new: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict response for new samples.

        Parameters:
            X_new: New feature matrix with the same number of features as training X.

        Returns:
            1-D ndarray of predicted values.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import plsr
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> y = pd.Series(np.random.rand(60))
            >>> model = plsr(X=X, y=y, n_components=2)
            >>> model.fit()
            >>> y_hat = model.predict(X)
        """
        if not isinstance(X_new, pd.DataFrame):
            X_new = pd.DataFrame(X_new, columns=self.features_name)
        X_scaled = self.scaler_.transform(X_new)
        return self.pls_model.predict(X_scaled).ravel()

    def get_scores(self) -> pd.DataFrame:
        """Return the T (X) scores DataFrame.

        Returns:
            DataFrame of shape (n_samples, n_components).

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import plsr
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> y = pd.Series(np.random.rand(60))
            >>> model = plsr(X=X, y=y, n_components=2)
            >>> model.fit()
            >>> df = model.get_scores()
        """
        return self.df_scores_

    def get_loadings(self) -> pd.DataFrame:
        """Return the P (X) loadings DataFrame.

        Returns:
            DataFrame of shape (n_features, n_components).

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import plsr
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> y = pd.Series(np.random.rand(60))
            >>> model = plsr(X=X, y=y, n_components=2)
            >>> model.fit()
            >>> df = model.get_loadings()
        """
        return self.df_loadings_

    def get_weights(self) -> pd.DataFrame:
        """Return the W (X) weights DataFrame.

        Returns:
            DataFrame of shape (n_features, n_components).

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import plsr
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> y = pd.Series(np.random.rand(60))
            >>> model = plsr(X=X, y=y, n_components=2)
            >>> model.fit()
            >>> df = model.get_weights()
        """
        return self.df_weights_

    def get_metrics(self) -> dict:
        """Return model performance metrics.

        Returns:
            dict with keys 'R2' (training), 'Q2' (LOO cross-validated), 'RMSE'.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import plsr
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> y = pd.Series(np.random.rand(60))
            >>> model = plsr(X=X, y=y, n_components=2)
            >>> model.fit()
            >>> m = model.get_metrics()
        """
        return self.metrics_

    def plot_predicted_vs_actual(
        self,
        fig_height: int = 600,
        fig_width: int = 700,
        font_size: int = 14,
    ) -> go.Figure:
        """Scatter plot of actual vs predicted response with R2 annotation.

        Parameters:
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Global font size.

        Returns:
            Plotly Figure object.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import plsr
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> y = pd.Series(np.random.rand(60))
            >>> model = plsr(X=X, y=y, n_components=2)
            >>> model.fit()
            >>> fig = model.plot_predicted_vs_actual()
            >>> fig.show()
        """
        y_actual = self.y
        y_pred = self.y_pred_
        r2 = self.metrics_['R2']
        q2 = self.metrics_['Q2']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_actual,
            y=y_pred,
            mode='markers',
            marker=dict(size=10, color='royalblue',
                        line=dict(width=1, color='DarkSlateGrey')),
            name='Samples',
        ))
        lim_min = float(min(y_actual.min(), y_pred.min()))
        lim_max = float(max(y_actual.max(), y_pred.max()))
        fig.add_trace(go.Scatter(
            x=[lim_min, lim_max],
            y=[lim_min, lim_max],
            mode='lines',
            line=dict(color='black', dash='dash', width=1),
            name='Ideal',
            showlegend=False,
        ))
        fig.add_annotation(
            x=0.05, y=0.95,
            xref='paper', yref='paper',
            text=f'R<sup>2</sup> = {r2:.3f}<br>Q<sup>2</sup> = {q2:.3f}',
            showarrow=False,
            font=dict(size=font_size),
            align='left',
        )
        fig.update_layout(
            title={'text': '<b>PLS-R: Predicted vs Actual</b>',
                   'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis_title='Actual',
            yaxis_title='Predicted',
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black',
                         zeroline=False)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black',
                         zeroline=False)
        return fig

    def plot_scores(
        self,
        fig_height: int = 700,
        fig_width: int = 900,
        font_size: int = 14,
    ) -> go.Figure:
        """Scatter plot of T1 vs T2 latent variable scores.

        Parameters:
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Global font size.

        Returns:
            Plotly Figure object.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import plsr
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> y = pd.Series(np.random.rand(60))
            >>> model = plsr(X=X, y=y, n_components=2)
            >>> model.fit()
            >>> fig = model.plot_scores()
            >>> fig.show()
        """
        df = self.df_scores_.copy()
        cols = df.columns.tolist()
        xc = cols[0] if len(cols) > 0 else 'LV1'
        yc = cols[1] if len(cols) > 1 else 'LV1'

        fig = px.scatter(
            df,
            x=xc,
            y=yc,
            color=pd.Series(self.y, name='y'),
            color_continuous_scale='Viridis',
            title='<b>PLS-R Scores Plot</b>',
            height=fig_height,
            width=fig_width,
        )
        fig.update_traces(
            marker=dict(size=12, opacity=0.85,
                        line=dict(width=1, color='DarkSlateGrey'))
        )
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black',
                         showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black',
                         showline=True, linewidth=2, linecolor='black')
        fig.update_layout(
            title={'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            font=dict(size=font_size),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig


# ---------------------------------------------------------------------------
# CLASS 3: ica
# ---------------------------------------------------------------------------

class ica:
    """Independent Component Analysis (ICA) for blind source separation.

    Parameters:
        X: Feature matrix (DataFrame or ndarray), rows=samples, cols=features.
        n_components: Number of independent components. Default is 2.
        max_iter: Maximum iterations for FastICA. Default is 1000.
        random_state: Random seed for reproducibility. Default is 42.
        scaling_method: One of "pareto", "mean", "uv", "minmax", or None.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.analysis.multivariate import ica
        >>> X = pd.DataFrame(np.random.rand(60, 50))
        >>> model = ica(X=X, n_components=2)
        >>> model.fit()
        >>> components = model.get_components()
        >>> fig = model.plot_components()
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        n_components: int = 2,
        max_iter: int = 1000,
        random_state: int = 42,
        scaling_method: str = 'pareto',
    ) -> None:
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError('X must be a DataFrame or ndarray')

        if isinstance(X, pd.DataFrame):
            self.features_name = list(X.columns)
        else:
            self.features_name = list(range(X.shape[1]))

        self.X = X
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaling_method = scaling_method

    def fit(self) -> None:
        """Fit the FastICA model to the scaled data.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import ica
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> model = ica(X=X, n_components=2)
            >>> model.fit()
        """
        X = self.X
        features_name = self.features_name

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=features_name)

        scale_power = _resolve_scale_power(self.scaling_method)
        scaler = Scaler(scale_power=scale_power)
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        ica_model = _FastICA(
            n_components=self.n_components,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        components = ica_model.fit_transform(X_scaled)
        mixing = ica_model.mixing_

        ic_cols = [f'IC{i+1}' for i in range(self.n_components)]

        df_components = pd.DataFrame(components, columns=ic_cols)
        df_mixing = pd.DataFrame(mixing, index=features_name, columns=ic_cols)

        self.ica_model = ica_model
        self.scaler_ = scaler
        self.df_components_ = df_components
        self.df_mixing_ = df_mixing

    def get_components(self) -> pd.DataFrame:
        """Return the IC component matrix (rows=samples).

        Returns:
            DataFrame of shape (n_samples, n_components).

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import ica
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> model = ica(X=X, n_components=2)
            >>> model.fit()
            >>> df = model.get_components()
        """
        return self.df_components_

    def get_mixing(self) -> pd.DataFrame:
        """Return the mixing matrix (rows=features).

        Returns:
            DataFrame of shape (n_features, n_components).

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import ica
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> model = ica(X=X, n_components=2)
            >>> model.fit()
            >>> df = model.get_mixing()
        """
        return self.df_mixing_

    def plot_components(
        self,
        ic: List[str] = ['IC1', 'IC2'],
        color_: Optional[Union[pd.Series, List[Any]]] = None,
        color_dict: Optional[dict] = None,
        fig_height: int = 900,
        fig_width: int = 1300,
        font_size: int = 20,
    ) -> go.Figure:
        """Scatter plot of two IC score components.

        Parameters:
            ic: Two IC component names for x and y axes.
            color_: Optional group labels for colouring points.
            color_dict: Optional mapping of group label to colour hex string.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Global font size.

        Returns:
            Plotly Figure object.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import ica
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> label = pd.Series(['A'] * 30 + ['B'] * 30)
            >>> model = ica(X=X, n_components=2)
            >>> model.fit()
            >>> fig = model.plot_components(ic=['IC1', 'IC2'], color_=label)
            >>> fig.show()
        """
        df = self.df_components_.copy()

        if color_ is not None:
            groups = color_.tolist() if isinstance(color_, pd.Series) else list(color_)
            df['Group'] = groups
            if color_dict is None:
                color_dict = _build_color_dict(groups)
            color_col = 'Group'
        else:
            color_col = None

        fig = px.scatter(
            df,
            x=ic[0],
            y=ic[1],
            color=color_col,
            color_discrete_map=color_dict,
            title=f'<b>ICA Components Plot</b> ({self.scaling_method} scaling)',
            height=fig_height,
            width=fig_width,
        )
        fig.update_traces(
            marker=dict(size=14, opacity=0.8,
                        line=dict(width=2, color='DarkSlateGrey'))
        )
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black',
                         showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black',
                         showline=True, linewidth=2, linecolor='black')
        fig.update_layout(
            title={'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            font=dict(size=font_size),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    def plot_mixing_(
        self,
        ic: List[str] = ['IC1', 'IC2'],
        fig_height: int = 600,
        fig_width: int = 1800,
        font_size: int = 20,
    ) -> go.Figure:
        """Plot the mixing matrix columns over features.

        Parameters:
            ic: IC column names to overlay.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Global font size.

        Returns:
            Plotly Figure object.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import ica
            >>> X = pd.DataFrame(np.random.rand(60, 50))
            >>> model = ica(X=X, n_components=2)
            >>> model.fit()
            >>> fig = model.plot_mixing_(ic=['IC1', 'IC2'])
            >>> fig.show()
        """
        df = self.df_mixing_.copy()
        df['Features'] = self.features_name

        fig = px.scatter(
            df,
            x='Features',
            y=ic,
            height=fig_height,
            width=fig_width,
            title='ICA Mixing Matrix',
        )
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
        fig.update_layout(
            title={'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            font=dict(size=font_size),
        )
        fig.update_traces(marker=dict(size=3))
        return fig


# ---------------------------------------------------------------------------
# CLASS 4: hca
# ---------------------------------------------------------------------------

class hca:
    """Hierarchical Cluster Analysis (HCA) with Plotly visualisation.

    Parameters:
        X: Feature matrix (DataFrame or ndarray), rows=samples, cols=features.
        label: Optional sample labels used for dendrogram leaf annotations.
        features_name: Optional feature names.
        method: Linkage method passed to scipy.cluster.hierarchy.linkage.
            Common values: "ward", "complete", "average", "single".
        metric: Distance metric passed to scipy.cluster.hierarchy.linkage.
        scaling_method: One of "pareto", "mean", "uv", "minmax", or None.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.analysis.multivariate import hca
        >>> X = pd.DataFrame(np.random.rand(30, 50))
        >>> label = pd.Series(['A'] * 15 + ['B'] * 15)
        >>> model = hca(X=X, label=label)
        >>> model.fit()
        >>> fig_dend = model.plot_dendrogram()
        >>> fig_heat = model.plot_heatmap(n_clusters=2)
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        label: Optional[Union[pd.Series, np.ndarray, List[Any]]] = None,
        features_name: Optional[Union[pd.Series, np.ndarray, List[Any]]] = None,
        method: str = 'ward',
        metric: str = 'euclidean',
        scaling_method: str = 'pareto',
    ) -> None:
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError('X must be a DataFrame or ndarray')

        if features_name is not None:
            if len(features_name) != X.shape[1]:
                raise ValueError('features_name length must match X.shape[1]')
            self.features_name = list(features_name)
        elif isinstance(X, pd.DataFrame):
            self.features_name = list(X.columns)
        else:
            self.features_name = list(range(X.shape[1]))

        n_samples = X.shape[0]
        if label is None:
            self.label = [f'S{i}' for i in range(n_samples)]
        else:
            self.label = list(label)

        self.X = X
        self.method = method
        self.metric = metric
        self.scaling_method = scaling_method

    def fit(self) -> None:
        """Compute the linkage matrix via hierarchical clustering.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import hca
            >>> X = pd.DataFrame(np.random.rand(30, 50))
            >>> model = hca(X=X)
            >>> model.fit()
        """
        X = self.X
        features_name = self.features_name

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=features_name)

        scale_power = _resolve_scale_power(self.scaling_method)
        scaler = Scaler(scale_power=scale_power)
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        Z = linkage(X_scaled, method=self.method, metric=self.metric)

        self.X_scaled_ = X_scaled
        self.linkage_matrix_ = Z

    def get_cluster_labels(self, n_clusters: int = 3) -> pd.Series:
        """Return flat cluster assignments via scipy fcluster.

        Parameters:
            n_clusters: Number of flat clusters to form.

        Returns:
            pd.Series of integer cluster IDs aligned to the original sample order.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import hca
            >>> X = pd.DataFrame(np.random.rand(30, 50))
            >>> model = hca(X=X)
            >>> model.fit()
            >>> clusters = model.get_cluster_labels(n_clusters=3)
        """
        labels = fcluster(self.linkage_matrix_, t=n_clusters, criterion='maxclust')
        return pd.Series(labels, name='Cluster', index=range(len(labels)))

    def plot_dendrogram(
        self,
        fig_height: int = 700,
        fig_width: int = 1200,
        font_size: int = 12,
        color_threshold: Optional[float] = None,
    ) -> go.Figure:
        """Draw the hierarchical dendrogram as a Plotly figure.

        The dendrogram is constructed by calling scipy.cluster.hierarchy.dendrogram
        and manually translating the coordinate output into Plotly line traces.

        Parameters:
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Tick label font size.
            color_threshold: Height threshold used to colour the dendrogram branches.
                Defaults to 70% of the maximum linkage height.

        Returns:
            Plotly Figure object.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import hca
            >>> X = pd.DataFrame(np.random.rand(30, 50))
            >>> label = [f'S{i}' for i in range(30)]
            >>> model = hca(X=X, label=label)
            >>> model.fit()
            >>> fig = model.plot_dendrogram()
            >>> fig.show()
        """
        Z = self.linkage_matrix_
        labels = self.label

        if color_threshold is None:
            color_threshold = 0.7 * float(Z[:, 2].max())

        dend = dendrogram(
            Z,
            labels=labels,
            color_threshold=color_threshold,
            no_plot=True,
        )

        # scipy returns matplotlib color codes ('C0','C1',...) — map to hex for Plotly
        _mpl_colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                       '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
        def _resolve_color(c):
            if isinstance(c, str) and len(c) == 2 and c[0] == 'C' and c[1].isdigit():
                return _mpl_colors[int(c[1]) % len(_mpl_colors)]
            return c

        fig = go.Figure()

        icoord = np.array(dend['icoord'])
        dcoord = np.array(dend['dcoord'])
        colors = [_resolve_color(c) for c in dend['color_list']]

        for xs, ys, color in zip(icoord, dcoord, colors):
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode='lines',
                line=dict(color=color, width=1.5),
                showlegend=False,
                hoverinfo='skip',
            ))

        leaf_positions = dend['leaves']
        leaf_xs = dend['icoord']
        tick_x = [np.mean(xs) for xs in leaf_xs]

        # Use scipy leaf positions to map tick labels
        ordered_labels = [labels[i] for i in dend['leaves']]
        # Compute correct x positions for leaves
        n = len(ordered_labels)
        leaf_x_vals = list(range(5, 10 * n + 5, 10))

        fig.update_layout(
            title={'text': '<b>Hierarchical Cluster Dendrogram</b>',
                   'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis=dict(
                tickvals=leaf_x_vals,
                ticktext=ordered_labels,
                tickangle=-90,
                tickfont=dict(size=font_size),
                showline=True,
                linewidth=1,
                linecolor='black',
            ),
            yaxis=dict(
                title='Distance',
                showline=True,
                linewidth=1,
                linecolor='black',
                tickfont=dict(size=font_size),
            ),
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    def plot_heatmap(
        self,
        n_clusters: int = 3,
        fig_height: int = 900,
        fig_width: int = 900,
        colorscale: str = 'RdBu_r',
        font_size: int = 12,
    ) -> go.Figure:
        """Clustered heatmap with rows ordered by hierarchical clustering.

        Parameters:
            n_clusters: Number of clusters for colour-bar annotation.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            colorscale: Plotly colorscale name for the heatmap.
            font_size: Global font size.

        Returns:
            Plotly Figure object.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.analysis.multivariate import hca
            >>> X = pd.DataFrame(np.random.rand(30, 50))
            >>> label = [f'S{i}' for i in range(30)]
            >>> model = hca(X=X, label=label)
            >>> model.fit()
            >>> fig = model.plot_heatmap(n_clusters=3)
            >>> fig.show()
        """
        Z = self.linkage_matrix_
        labels = self.label
        features_name = self.features_name
        X_scaled = self.X_scaled_

        # Row ordering from dendrogram
        dend = dendrogram(Z, no_plot=True)
        row_order = dend['leaves']

        X_ordered = X_scaled[row_order, :]
        labels_ordered = [labels[i] for i in row_order]

        # Cluster assignments for colour bar
        cluster_ids = fcluster(Z, t=n_clusters, criterion='maxclust')
        cluster_ordered = cluster_ids[row_order]

        cluster_palette = _build_color_dict(list(range(1, n_clusters + 1)))

        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=X_ordered,
            x=list(map(str, features_name)),
            y=labels_ordered,
            colorscale=colorscale,
            colorbar=dict(title='Value', len=0.85),
        ))

        # Add cluster annotation as a narrow heatmap strip on the left
        cluster_colors_numeric = cluster_ordered.tolist()
        fig.add_trace(go.Heatmap(
            z=[[v] for v in cluster_colors_numeric],
            x=['Cluster'],
            y=labels_ordered,
            colorscale=[
                [i / max(1, n_clusters - 1), cluster_palette.get(i + 1, '#999999')]
                for i in range(n_clusters)
            ],
            showscale=False,
            xaxis='x2',
        ))

        fig.update_layout(
            title={'text': '<b>Clustered Heatmap</b>',
                   'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            xaxis=dict(showticklabels=False, domain=[0.08, 1.0]),
            xaxis2=dict(domain=[0.0, 0.06], anchor='y'),
            yaxis=dict(tickfont=dict(size=font_size)),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        return fig
