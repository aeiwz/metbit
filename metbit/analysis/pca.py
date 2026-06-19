# -*- coding: utf-8 -*-

__author__ = 'aeiwz'

from typing import Optional, List, Union, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as _PCA
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from ..preprocessing.scaler_ext import Scaler
from ..viz.ellipse import confidence_ellipse
from .opls_da import _resolve_scale_power


class pca:

    '''

    PCA model

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    label : array-like, shape (n_samples,)
        Target data, where n_samples is the number of samples.
    features_name : array-like, shape (n_features,), default=None
        Name of features.
    n_components : int, default=2
        Number of components to keep.
    scale : str, default='pareto'
        Method of scaling. 'pareto' for pareto scaling, 'mean' for mean centering, 'uv' for unitvarian scaling.
    random_state : int, default=42
        Random state for permutation test.
    test_size : float, default=0.3
        Size of test set.

    Examples:
    ----------
    import pandas as pd
    import numpy as np
    from metbit import pca

    # Create a dataset
    data = pd.DataFrame(np.random.rand(500, 50000))
    class_ = pd.Series(np.random.choice(['A', 'B', 'C'], 500), name='Group')
    time = pd.Series(np.random.choice(['1-wk', '2-wk', '3-wk', '4-wk'], 500), name='Time point')


    # Assign X and target
    X = datasets.iloc[:, 2:]
    y = datasets['Group']
    time = datasets['Time point']
    features_name = list(X.columns.astype(float))

    ## Perform PCA model


    pca_mod = pca(X = X, label = y, features_name=features_name, n_components=2, scaling_method='pareto', random_state=42, test_size=0.3)
    pca_mod.fit()


    # Visualisation of PCA model
    pca_mod.plot_observe_variance()

    pca_mod.plot_cumulative_observed()

    shape_ = {'1-wk': 'circle', '2-wk': 'square', '3-wk': 'diamond', '4-wk': 'cross'}

    pca_mod.plot_pca_scores(symbol=time, symbol_dict=shape_)

    pca_mod.plot_loading_()

    pca_mod.plot_pca_trajectory(time_=time, time_order={'1-wk': 0, '2-wk': 1, '3-wk': 2, '4-wk': 3}, color_dict={'A': '#636EFA', 'B': '#EF553B', 'C': '#00CC96'}, symbol_dict=shape_)

    '''

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        label: Optional[Union[pd.Series, np.ndarray, List[Any]]] = None,
        features_name: Optional[Union[pd.Series, np.ndarray, List[Any]]] = None,
        n_components: int = 2,
        scaling_method: str = 'pareto',
        random_state: int = 42,
        test_size: float = 0.3,
    ) -> None:




        if features_name is not None:
            if not isinstance(features_name, (pd.Series, np.ndarray, list)):
                raise ValueError('features_name must be a series, list or 1D array')
            if len(features_name) != X.shape[1]:
                raise ValueError('features_name must have the same number of features as X')

        if label is None:
            label = ["data" for x in range(X.shape[0])]

        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError('X must be a dataframe or array')

        if not isinstance(n_components, int):
            raise ValueError('n_components must be an integer')

        if not isinstance(scaling_method, str):
            raise ValueError('scaling method must be a string')

        if not isinstance(random_state, int):
            raise ValueError('random_state must be an integer')

        if not isinstance(label, (pd.Series, np.ndarray, list)):
            raise ValueError('label must be a series, list or array')
        if len(label) != X.shape[0]:
            raise ValueError('X and label must have the same number of samples')

        self.features_name = features_name
        if features_name is None:
            if isinstance(X, pd.DataFrame):
                self.features_name = X.columns
            else:
                self.features_name = np.arange(X.shape[1])
        else:
            self.features_name = features_name


        # Check missing values in X
        if isinstance(X, pd.DataFrame):
            if X.isnull().sum().sum() > 0:
                raise ValueError('X contains missing values')
        else:
            if np.isnan(X).sum().sum() > 0:
                raise ValueError('X contains missing values')

        self.X = X
        self.label = label
        self.n_components = n_components
        self.scaling_method = scaling_method
        self.random_state = random_state


        self.test_size = test_size



    def fit(self):
        test_size=self.test_size

        X = self.X
        label = self.label
        n_components = self.n_components
        scaling_method = self.scaling_method
        random_state = self.random_state
        features_name = self.features_name
        Y = pd.Categorical(label).codes


        if isinstance(features_name, list):
            features_name = list(features_name)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=features_name)

        if not isinstance(label, pd.Series):
            label = pd.Series(label)

        Scale_power = _resolve_scale_power(scaling_method)

        model_scaler = Scaler(scale_power=Scale_power)
        model_scaler.fit(X)
        model_X = model_scaler.transform(X)



        pca_model = _PCA(n_components=n_components)
        pca_model.fit(model_X)

        self.scores_ = pca_model.transform(model_X)
        self.loadings_ = pca_model.components_.T


        #Create dataframe for scores depending on the number of components
        for i in range(n_components):
            if i == 0:
                df_scores_ = pd.DataFrame(self.scores_[:,i], columns=['PC{}'.format(i+1)])
            else:
                df_scores_['PC{}'.format(i+1)] = self.scores_[:,i]
        df_scores_.index = label.index

        df_scores_['Group'] = label

        self.df_scores_ = df_scores_

        #Create dataframe for loadings depending on the number of components
        for i in range(n_components):
            if i == 0:
                df_loadings_ = pd.DataFrame(self.loadings_[:,i], index=features_name, columns=['PC{}'.format(i+1)])
            else:
                df_loadings_['PC{}'.format(i+1)] = self.loadings_[:,i]

        df_loadings_['Features'] = features_name

        self.df_loadings_ = df_loadings_

        explained_variance_ = pca_model.explained_variance_ratio_
        explained_variance_ = np.insert(explained_variance_, 0, 0)
        cumulative_variance_ = np.cumsum(explained_variance_)


        r2_index = ['']
        for i in range(n_components):
            r2_index.append('PC{}'.format(i+1))

        df_explained_variance_ = pd.DataFrame(r2_index, columns=['PC'])
        df_explained_variance_['Explained variance'] = explained_variance_
        df_explained_variance_['Cumulative variance'] = cumulative_variance_



        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        X_test = model_scaler.transform(X_test)
        X_test_pca = pca_model.transform(X_test)

            # Inverse transform the test set from the PCA space
        X_test_reconstructed = pca_model.inverse_transform(X_test_pca)


        # Calculate Q2 score for the test set
        q2_test = r2_score(X_test, X_test_reconstructed)


        self.q2_test = q2_test
        self.explained_variance_ = explained_variance_
        self.cumulative_variance_ = cumulative_variance_
        self.df_explained_variance_ = df_explained_variance_
        self.pca_model = pca_model
        self.model_scaler = model_scaler
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_reconstructed = X_test_reconstructed
        self.X_test_pca = X_test_pca
        self.df_scores_ = df_scores_
        self.df_loadings_ = df_loadings_

        return pca_model

    def get_explained_variance(self) -> pd.DataFrame:
        df_explained_variance_ = self.df_explained_variance_
        return df_explained_variance_

    def get_scores(self) -> pd.DataFrame:
        df_scores_ = self.df_scores_
        return df_scores_

    def get_loadings(self) -> pd.DataFrame:
        df_loadings_ = self.df_loadings_
        return df_loadings_

    def get_q2_test(self) -> float:
        q2_test = self.q2_test
        return q2_test

    def plot_observe_variance(self, fig_height: int = 600, fig_width: int = 800, font_size: int = 15) -> go.Figure:

        '''
        Visualise explained variance plot

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Explained variance plot.

        '''

        scaling_method = self.scaling_method

        df_explained_variance_ = self.df_explained_variance_

        fig = px.bar(df_explained_variance_,
                x='PC', y=df_explained_variance_['Explained variance'],
                text=np.round(df_explained_variance_['Explained variance'], decimals=3),
                width=fig_width, height=fig_height,
                title='Explained Variance ({} scaling)'.format(scaling_method))
        fig.update_layout(
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            font=dict(size=font_size))
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        return fig


    def plot_cumulative_observed(self, fig_height: int = 600, fig_width: int = 800, font_size: int = 15, marker_size: int = 10) -> go.Figure:

        '''
        Visualise cumulative variance plot

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Cumulative variance plot.

        '''

        df_explained_variance_ = self.df_explained_variance_

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_explained_variance_['PC'],
                y=df_explained_variance_['Cumulative variance'],
                marker=dict(size=marker_size, color="LightSeaGreen"),
                name='R<sup>2</sup>X (Cum)'
            ))

        fig.add_trace(
            go.Bar(
                x=df_explained_variance_['PC'],
                y=df_explained_variance_['Explained variance'],
                marker=dict(color="RoyalBlue"),
                name='R<sup>2</sup>X',
                text=np.round(df_explained_variance_['Explained variance'], decimals=3)
            ))
        fig.update_layout(width=fig_width, height=fig_height,
                        title='Explained Variance and Cumulative Variance')
        fig.update_layout(
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            font=dict(size=font_size))

        return fig



    def plot_pca_scores(self, pc: List[str] = ['PC1', 'PC2'],
                        color_: Optional[pd.Series] = None, color_dict: Optional[dict] = None,
                        symbol_: Optional[pd.Series] = None, symbol_dict: Optional[dict] = None,
                        marker_label: Optional[pd.Series] = None,
                        fig_height: int = 900, fig_width: int = 1300,
                        marker_size: int = 35, marker_opacity: float = 0.7,
                        font_size: int = 20, title_font_size: int = 21,
                        individual_ellipse: bool = True,
                        legend_name: List[str] = ['Group', 'Time point']) -> go.Figure:

        '''
        Visualise PCA scores plot

        Parameters
        ----------
        pc : list, default=['PC1', 'PC2']
            List of principal components to plot.
        color: array-like, shape (n_samples,), default=None
            Target data, where n_samples is the number of samples.
        color_dict : dict, default=None
            Dictionary of color_ mapping.
        symbol_ : array-like, shape (n_samples,), default=None
            Target data, where n_samples is the number of samples.
        symbol_dict : dict, default=None
            Dictionary of symbol_ mapping.
        fig_height : int, default=900
            Height of figure.
        fig_width : int, default=1300
            Width of figure.
        marker_size : int, default=35
            Size of marker.
        marker_opacity : float, default=0.7
            Opacity of marker.
        text_ : array-like, shape (n_samples,), default=None
            Text to display on each point.


        Returns
        -------
        fig : plotly.graph_objects.Figure
            PCA scores plot.

        '''

        scaling_method = self.scaling_method
        df_scores_ = self.df_scores_
        r2 = self.df_explained_variance_
        q2_test = self.q2_test

        if color_ is not None:
            if len(color_) != len(self.label):
                raise ValueError('color_ must have the same number of samples as y')
            else:
                color_ = color_


        if symbol_ is not None:
            if len(symbol_) != len(self.label):
                raise ValueError('symbol_ must have the same number of samples as y')

        if color_ is not None:
            df_scores_['Group'] = color_
        else:
            pass

        #check symbol_ dimension must be equal to y
        if symbol_ is not None:
            if len(symbol_) != len(self.label):
                raise ValueError('symbol_ must have the same number of samples as y')  # pragma: no cover

        #check symbol_dict must be a dictionary
        if symbol_dict is not None:
            if not isinstance(symbol_dict, dict):
                raise ValueError('symbol_dict must be a dictionary')
        else:
            symbol_dict = None


        # pc must be a list of 2
        if not isinstance(pc, list):
            raise ValueError("pc must be a list of string \n Example: pc=['PC1', 'PC2']")
        if len(pc) != 2:
            raise ValueError('pc must be a list of 2')
        # pc must be match with columns of df_scores_
        if pc[0] not in self.df_scores_.columns:
            raise ValueError("pc must be in df_scores_ columns \n Example: pc=['PC1', 'PC2']")
        if pc[1] not in self.df_scores_.columns:
            raise ValueError("pc must be in df_scores_ columns \n Example: pc=[\'PC1\', \'PC2\']")



        r2 = self.df_explained_variance_
        q2_test = self.q2_test
        df_scores_['Index'] = df_scores_.index


        #If user not input color_dict then get unique of label and create color_dict
        if color_dict is not None:
            color_dict_2 = color_dict
        else:

            import plotly.colors as plotly_colour

            name_color_set = ['Plotly', 'D3', 'G10', 'T10', 'Alphabet', 'Dark24', 'Light24', 'Set1', 'Pastel1',
                                'Dark2', 'Set2', 'Pastel2', 'Set3', 'Antique', 'Safe', 'Bold', 'Pastel',
                                'Vivid', 'Prism']

            palette = []
            for name in name_color_set:
                palette += getattr(plotly_colour.qualitative, name) # This is a list of colors

            color_dict = {i: palette[i] for i in range(len(df_scores_['Group'].unique()))}

            group_unique = df_scores_['Group'].unique()
            color_dict_2 = {group_unique[i]: list(color_dict.values())[i] for i in range(len(group_unique))}


        fig = px.scatter(df_scores_, x=pc[0], y=pc[1], color='Group',
                        symbol=symbol_,
                        color_discrete_map=color_dict_2,
                        symbol_map=symbol_dict,
                        text=marker_label,
                        title=f'<b>PCA Scores Plot<b> {self.scaling_method} scaling',
                        height=fig_height, width=fig_width,
                        labels={'color': legend_name[0], 'symbol': legend_name[1],
                                'Group': legend_name[0],
                                pc[0]: "{} R<sup>2</sup>X: {}%".format(pc[0], np.round(r2.loc[r2.loc[r2['PC']==pc[0]].index, 'Explained variance'].values[0]*100, decimals=2)),
                                pc[1]: "{} R<sup>2</sup>X: {}%".format(pc[1], np.round(r2.loc[r2.loc[r2['PC']==pc[1]].index, 'Explained variance'].values[0]*100, decimals=2))},
                        hover_data={'Group':True, 'Index':True, pc[0]:True, pc[1]:True})


        fig.update_traces(marker=dict(size=marker_size,
                            opacity=marker_opacity,
                            line=dict(width=2, color='DarkSlateGrey')))

        fig.update_traces(textposition='middle center',
                            textfont_size=marker_size-(0.4*marker_size))


        fig.add_annotation(dict(font=dict(color="black",size=font_size),
                                #x=x_loc,
                                x=1.0,
                                y=0.05,
                                showarrow=False,
                                text=f"<b>R<sup>2</sup>X (Cum): {np.round(r2.loc[r2.loc[r2['PC']==pc[1]].index, 'Cumulative variance'].values[0]*100, decimals=2)}%<b>",
                                textangle=0,
                                xref="paper",
                                yref="paper"),
                                # set alignment of text to left side of entry
                                align="left")

        fig.add_annotation(dict(font=dict(color="black",size=font_size),
                                #x=x_loc,
                                x=1.0,
                                y=0.01,
                                showarrow=False,
                                text=f"<b>Q<sup>2</sup>X (Cum): {np.round(q2_test*100, decimals=2)}%<b>",
                                textangle=0,
                                xref="paper",
                                yref="paper"),
                                # set alignment of text to left side of entry
                                align="left")

        if individual_ellipse == True:
            for circle_ in df_scores_['Group'].unique():

                fig.add_shape(type='path',
                    path=confidence_ellipse(df_scores_[df_scores_['Group']==circle_][pc[0]], df_scores_[df_scores_['Group']==circle_][pc[1]]),
                    line=dict(color=color_dict_2[circle_], width=2))
        else:
            fig.add_shape(type='path',
                    path=confidence_ellipse(df_scores_[pc[0]], df_scores_[pc[1]]))



        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_layout(
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            font=dict(size=title_font_size))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

        return fig



    def plot_loading_(self, pc: List[str] = ['PC1', 'PC2'], fig_height: int = 600, fig_width: int = 1800,
                      font_size: int = 20, title_font_size: int = 20, marker_size: int = 1,
                      x_axis_title: str = '𝛿<sub>H</sub> in ppm', xaxis_direction: str = "reversed") -> go.Figure:

        '''
        Visualise PCA loadings

        Parameters
        ----------
        pc : list, default=['PC1', 'PC2']
            Principle component to plot.
        fig_height : int, default=600
            Height of figure.
        fig_width : int, default=1800
            Width of figure.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Plotly figure.

        ----------
        '''
        pc = pc
        df_loadings_ = self.df_loadings_

        loadings_label = self.features_name
        df_loadings_['Features'] = loadings_label


        fig = px.scatter(df_loadings_, x='Features', y=pc,
                                height=fig_height, width=fig_width,
                                title='Loadings plot')

        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
        fig.update_layout(title={'y':0.95,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'},
                        font=dict(size=title_font_size))

        fig.update_layout(xaxis = dict(autorange=xaxis_direction))

        fig.update_traces(marker=dict(size=marker_size))
        fig.update_layout(xaxis_title=x_axis_title)


        return fig




    def plot_pca_trajectory(self, time_: pd.Series, time_order: dict, stat_: List[str] = ['mean', 'sem'], pc: List[str] = ['PC1', 'PC2'],
                            color_dict: Optional[dict] = None, symbol_dict: Optional[dict] = None,
                            fig_height: int = 900, fig_width: int = 1300,
                            marker_size: int = 35, marker_opacity: float = 0.7,
                            title_font_size: int = 20, font_size: int = 20,
                            legend_name: List[str] = ['Group', 'Time point']) -> go.Figure:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
        import numpy as np

        if not isinstance(time_order, dict):
            raise ValueError("`time_order` must be a dictionary.")
        if time_ is None or len(time_) != len(self.label):
            raise ValueError("`time_` must match the length of label and not be None.")
        if stat_[0] not in ['mean', 'median'] or stat_[1] not in ['sem', 'std']:
            raise ValueError("`stat_` must be ['mean' or 'median', 'sem' or 'std'].")

        # Copy scores and add time point
        df_scores_ = self.df_scores_.copy()
        df_scores_['Time point'] = time_

        # Aggregate means/medians
        if stat_[0] == 'mean':
            df_scores_point = df_scores_.groupby(['Group', 'Time point']).mean()
        else:
            df_scores_point = df_scores_.groupby(['Group', 'Time point']).median()

        # Aggregate errors
        if stat_[1] == 'sem':
            err_df = df_scores_.groupby(['Group', 'Time point']).sem()
        else:
            err_df = df_scores_.groupby(['Group', 'Time point']).std()

        # Map order for sorting
        df_scores_point['Time order'] = df_scores_point.index.get_level_values('Time point').map(time_order)
        err_df['Time order'] = err_df.index.get_level_values('Time point').map(time_order)

        df_scores_point.reset_index(inplace=True)
        err_df.reset_index(inplace=True)

        df_scores_point.sort_values(by=['Group', 'Time order'], inplace=True)
        err_df.sort_values(by=['Group', 'Time order'], inplace=True)

        # Default color palette if not provided
        if color_dict is None:
            import plotly.colors as plotly_color
            palette = sum([getattr(plotly_color.qualitative, name) for name in ['Plotly', 'D3', 'Set2']], [])
            groups = df_scores_['Group'].unique()
            color_dict = {group: palette[i % len(palette)] for i, group in enumerate(groups)}

        if symbol_dict is None:
            symbol_dict = {}

        r2 = self.df_explained_variance_
        q2_test = self.q2_test

        # Main line+error plot
        fig = px.line(df_scores_point, x=pc[0], y=pc[1], line_group='Group',
                    error_x=err_df[pc[0]], error_y=err_df[pc[1]],
                    color='Group', color_discrete_map=color_dict,
                    symbol='Time point', symbol_map=symbol_dict,
                    title=f'<b>Principle component analysis ({self.scaling_method})<b>',
                    height=fig_height, width=fig_width,
                    labels={
                        pc[0]: f"{pc[0]} R<sup>2</sup>X: {np.round(r2.loc[r2['PC'] == pc[0], 'Explained variance'].values[0]*100, 2)} %",
                        pc[1]: f"{pc[1]} R<sup>2</sup>X: {np.round(r2.loc[r2['PC'] == pc[1], 'Explained variance'].values[0]*100, 2)} %",
                        'Group': legend_name[0],
                        'Time point': legend_name[1]
                    })

        # Connect lines
        for group in df_scores_point['Group'].unique():
            df_group = df_scores_point[df_scores_point['Group'] == group]
            fig.add_trace(go.Scatter(
                x=df_group[pc[0]], y=df_group[pc[1]],
                mode='lines',
                line=dict(color=color_dict[group], width=2),
                showlegend=False
            ))

        # Ellipse for global trajectory center
        fig.add_shape(type='path',
                    path=confidence_ellipse(df_scores_point[pc[0]], df_scores_point[pc[1]]),
                    line=dict(color='black', width=2))

        # Axis and annotations
        fig.update_xaxes(tickformat=".1e", zeroline=True, zerolinewidth=2, zerolinecolor='Black', showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(tickformat=".1e", zeroline=True, zerolinewidth=2, zerolinecolor='Black', showline=True, linewidth=2, linecolor='black')
        fig.update_layout(
            title={'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            font=dict(size=title_font_size),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig.update_traces(marker=dict(size=marker_size, opacity=marker_opacity, line=dict(width=2, color='DarkSlateGrey')))

        fig.add_annotation(dict(font=dict(color="black", size=font_size),
                                x=1.0, y=0.05, showarrow=False,
                                text=f"<b>R<sup>2</sup>X (Cum): {np.round(r2.loc[r2['PC'] == pc[1], 'Cumulative variance'].values[0]*100, 2)}%<b>",
                                xref="paper", yref="paper"))
        fig.add_annotation(dict(font=dict(color="black", size=font_size),
                                x=1.0, y=0.01, showarrow=False,
                                text=f"<b>Q<sup>2</sup>X (Cum): {np.round(q2_test*100, 2)}%<b>",
                                xref="paper", yref="paper"))

        return fig


    def plot_3d_pca(self, pc: List[str] = ['PC1', 'PC2', 'PC3'], color_: Optional[pd.Series] = None, color_dict: Optional[dict] = None,
                    symbol_: Optional[pd.Series] = None, symbol_dict: Optional[dict] = None, fig_height: int = 900, fig_width: int = 1300,
                    marker_size: int = 35, marker_opacity: float = 0.7, marker_label: Optional[pd.Series] = None, font_size: int = 20, title_font_size: int = 20,
                    legend_name: List[str] = ['Group', 'Time point']) -> go.Figure:
        import plotly.express as px
        '''
        Visualise 3D PCA scores plot

        Parameters
        ----------
        pc : list, default=['PC1', 'PC2', 'PC3']
            List of principal components to plot.
        color: array-like, shape (n_samples,), default=None
            Target data, where n_samples is the number of samples.
        color_dict : dict, default=None
            Dictionary of color_ mapping.
        symbol_ : array-like, shape (n_samples,), default=None
            Target data, where n_samples is the number of samples.
        symbol_dict : dict, default=None
            Dictionary of symbol_ mapping.
        fig_height : int, default=900
            Height of figure.
        fig_width : int, default=1300
            Width of figure.
        marker_size : int, default=35
            Size of marker.
        marker_opacity : float, default=0.7
            Opacity of marker.
        text_ : array-like, shape (n_samples,), default=None
            Text to display on each point.


        Returns
        -------
        fig : plotly.graph_objects.Figure
            PCA scores plot.

        '''



        scaling_method = self.scaling_method
        df_scores_ = self.df_scores_
        r2 = self.df_explained_variance_
        q2_test = self.q2_test

        if color_ is not None:
            if len(color_) != len(self.label):
                raise ValueError('color_ must have the same number of samples as y')
            else:
                color_ = color_


        if symbol_ is not None:
            if len(symbol_) != len(self.label):
                raise ValueError('symbol_ must have the same number of samples as y')

        if color_ is not None:
            df_scores_['Group'] = color_
        else:
            pass

        #check symbol_ dimension must be equal to y
        if symbol_ is not None:
            if len(symbol_) != len(self.label):
                raise ValueError('symbol_ must have the same number of samples as y')  # pragma: no cover

        #check symbol_dict must be a dictionary
        if symbol_dict is not None:
            if not isinstance(symbol_dict, dict):
                raise ValueError('symbol_dict must be a dictionary')
        else:
            symbol_dict = None


        # pc must be a list of 3
        if not isinstance(pc, list):
            raise ValueError("pc must be a list of string \n Example: pc=['PC1', 'PC2', 'PC3']")
        if len(pc) != 3:
            raise ValueError('pc must be a list of 3')
        # pc must be match with columns of df_scores_
        if pc[0] not in self.df_scores_.columns:
            raise ValueError("pc must be in df_scores_ columns \n Example: pc=['PC1', 'PC2', 'PC3']")
        if pc[1] not in self.df_scores_.columns:
            raise ValueError("pc must be in df_scores_ columns \n Example: pc=[\'PC1\', \'PC2\', \'PC3\']")
        if pc[2] not in self.df_scores_.columns:
            raise ValueError("pc must be in df_scores_ columns \n Example: pc=[\'PC1\', \'PC2\', \'PC3\']")

        r2 = self.df_explained_variance_
        q2_test = self.q2_test
        df_scores_['Index'] = df_scores_.index


        #If user not input color_dict then get unique of label and create color_dict
        if color_dict is not None:
            color_dict_2 = color_dict
        else:

            import plotly.colors as plotly_colour

            name_color_set = ['Plotly', 'D3', 'G10', 'T10', 'Alphabet', 'Dark24', 'Light24', 'Set1', 'Pastel1',
                                'Dark2', 'Set2', 'Pastel2', 'Set3', 'Antique', 'Safe', 'Bold', 'Pastel',
                                'Vivid', 'Prism']

            palette = []
            for name in name_color_set:
                palette += getattr(plotly_colour.qualitative, name) # This is a list of colors

            color_dict = {i: palette[i] for i in range(len(df_scores_['Group'].unique()))}

            group_unique = df_scores_['Group'].unique()
            color_dict_2 = {group_unique[i]: list(color_dict.values())[i] for i in range(len(group_unique))}



        fig = px.scatter_3d(df_scores_, x=pc[0], y=pc[1], z=pc[2], color='Group', symbol=symbol_,
                            color_discrete_map=color_dict_2, symbol_map=symbol_dict,
                            text=marker_label,
                            title=f'<b>PCA Scores Plot<b> {self.scaling_method} scaling',
                            height=fig_height, width=fig_width,
                            labels={'color': legend_name[0], 'symbol': legend_name[1],
                                    'Group': legend_name[0],
                                    pc[0]: "{} R<sup>2</sup>X: {}%".format(pc[0], np.round(r2.loc[r2.loc[r2['PC']==pc[0]].index, 'Explained variance'].values[0]*100, decimals=2)),
                                    pc[1]: "{} R<sup>2</sup>X: {}%".format(pc[1], np.round(r2.loc[r2.loc[r2['PC']==pc[1]].index, 'Explained variance'].values[0]*100, decimals=2)),
                                    pc[2]: "{} R<sup>2</sup>X: {}%".format(pc[2], np.round(r2.loc[r2.loc[r2['PC']==pc[2]].index, 'Explained variance'].values[0]*100, decimals=2))},
                            hover_data={'Group':True, 'Index':True, pc[0]:True, pc[1]:True, pc[2]:True})

        fig.update_traces(marker=dict(size=marker_size,
                            opacity=marker_opacity,
                            line=dict(width=2, color='DarkSlateGrey')))

        fig.update_traces(textposition='middle center',
                    textfont_size=marker_size-(0.4*marker_size))

        fig.add_annotation(dict(font=dict(color="black",size=font_size), x=1.0, y=0.05, showarrow=False,
                                text=f"<b>R<sup>2</sup>X (Cum): {np.round(r2.loc[r2.loc[r2['PC']==pc[2]].index, 'Cumulative variance'].values[0]*100, decimals=2)}%<b>",
                                textangle=0, xref="paper", yref="paper"), align="left")

        fig.add_annotation(dict(font=dict(color="black",size=font_size), x=1.0, y=0.01, showarrow=False,
                                text=f"<b>Q<sup>2</sup>X (Cum): {np.round(q2_test*100, decimals=2)}%<b>",
                                textangle=0, xref="paper", yref="paper"), align="left")

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

        return fig
