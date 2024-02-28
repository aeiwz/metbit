# -*- coding: utf-8 -*-


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import permutation_test_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import opls.cross_validation
import opls.plotting
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

from pyChemometrics import ChemometricsScaler

import os

__auther__ = "aeiwz"


class opls_da:
    
    
        
    # Import necessary libraries
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import permutation_test_score
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    from sklearn.utils import shuffle
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    import opls.cross_validation
    import opls.plotting
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    import warnings
    warnings.filterwarnings('ignore')

    from pyChemometrics import ChemometricsScaler

    import os




    '''
    OPLS-DA model
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y : array-like, shape (n_samples,)
        Target data, where n_samples is the number of samples.
    n_components : int, default=2
        Number of components to keep.
    scale : str, default='par'
        Method of scaling. 'par' for pareto scaling, 'mc' for mean centering, 'uv' for unitvarian scaling.
    cv : int, default=5
        Number of cross-validation folds.
    n_permutations : int, default=1000
        Number of permutations for permutation test.
    random_state : int, default=42
        Random state for permutation test.
    kfold : int, default=3
        Number of cross-validation folds.
        

    '''
    

        
    
    def __init__(self, X, y,features_name=None, n_components=2, scale='pareto', kfold=3, estimator='opls', random_state=42):
        


        #check X and y must be dataframe or array
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError('X must be a dataframe or array')
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError('y must be a series or array')
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of samples')
        if not isinstance(n_components, int):
            raise ValueError('n_components must be an integer')
        if not isinstance(scale, str):
            raise ValueError('scale must be a string')
        if not isinstance(kfold, int):
            raise ValueError('kfold must be an integer')
        if not isinstance(estimator, str):
            raise ValueError('estimator must be a string')
        if not isinstance(random_state, int):
            raise ValueError('random_state must be an integer')
        if features_name is not None:
            if not isinstance(features_name, (pd.Series, np.ndarray, list)):
                raise ValueError('features_name must be a series, list or 1D array')
            if len(features_name) != X.shape[1]:
                raise ValueError('features_name must have the same number of features as X')
            
            
            
        #check unique values in y
        if isinstance(y, pd.Series):
            if len(y.unique()) < 2:
                raise ValueError('OPLS-DA requires at least 2 group comparisons')
        if isinstance(y, np.ndarray):
            if len(np.unique(y)) < 2:
                raise ValueError('OPLS-DA requires at least 2 group comparisons')
        if isinstance(y, list):
            if len(np.unique(y)) < 2:
                raise ValueError('OPLS-DA requires at least 2 group comparisons')        
            
        #check unique values in y
        if isinstance(y, pd.Series):
            if len(y.unique()) > 2:
                raise ValueError('OPLS-DA requires only 2 group comparisons')
        if isinstance(y, np.ndarray):
            if len(np.unique(y)) < 2:
                raise ValueError('OPLS-DA requires only 2 group comparisons')
        if isinstance(y, list):
            if len(np.unique(y)) < 2:
                raise ValueError('OPLS-DA requires only 2 group comparisons')



        self.features_name = features_name
        if features_name is None:
            if isinstance(X, pd.DataFrame):
                self.features_name = X.columns
            else:
                self.features_name = np.arange(X.shape[1])
        else:
            self.features_name = features_name
            
        if isinstance(X, pd.DataFrame):
            self.X = X.values
        else:
            self.X = X   
                     
        self.X = X
        self.y = y
        self.n_components = n_components
        self.scale = scale
        self.random_state = random_state
        self.opls_model = None
        self.opls_cv = None
        self.opls_permutation_cv = None
        self.opls_permutation_cv_scores = None
        self.opls_permutation_cv_score = None
        self.opls_permutation_cv_score_std = None
        self.kfold = kfold
        self.estimator = estimator
        
        
    def fit(self):
        
        X = self.X
        y = self.y
        n_components = self.n_components
        scale = self.scale
        cv = self.cv
        n_permutations = self.n_permutations
        random_state = self.random_state
        kfold = self.kfold
        estimator = self.estimator
        
        if scale == 'pareto':
            scale_power = 0.5
        elif scale == 'mean':
            scale_power = 0
        elif scale == 'uv':
            scale_power = 1
        elif scale == 'minmax':
            scale_power = 0
            
        self.scale = scale
            
            
        # Create a pipeline with data preprocessing and OPLS-DA model
        pipeline = Pipeline([
                                ('scale', ChemometricsScaler(scale_power=scale_power)),
                                ('oplsda', PLSRegression(n_components=n_components)),
                                ('opls', opls.cross_validation.CrossValidation(kfold=kfold, estimator=estimator, scaler=scale))
                            ])

        oplsda = pipeline.named_steps['oplsda']
        cv = pipeline.named_steps['opls']
        cv.fit(X, y)

        oplsda.fit(X, pd.Categorical(y).codes)
        
        s_scores_df = pd.DataFrame({'correlation': cv.correlation,'covariance': cv.covariance}, index=features_name)
        df_opls_scores = pd.DataFrame({'t_scores': cv.scores, 't_ortho': cv.orthogonal_score, 't_pred': cv.predictive_score, 'label': y})

        self.s_scores_df = s_scores_df
        self.df_opls_scores = df_opls_scores
        
        self.oplsda = oplsda
        self.cv = cv
    
        return oplsda, cv

    def permutation_test(self, n_permutations=500, cv=3, n_jobs=-1, verbose=10):
        
        
        self.cv = cv
        self.n_permutations = n_permutations
        oplsda = self.oplsda
        X = self.X
        y = self.y
        pipeline = self.pipeline
        randomstate = self.random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        oplsda.fit(X, pd.Categorical(y).codes)

        # Permutation test to assess the significance of the model
        acc_score, permutation_scores, p_value = permutation_test_score(
        pipeline.named_steps['oplsda'], X, pd.Categorical(y).codes, cv=3, n_permutations=n_permutations, n_jobs=n_jobs, random_state=randomstate, verbose=verbose)


        self.acc_score = acc_score
        self.permutation_scores = permutation_scores
        self.p_value = p_value
        
        
    
    def vip_scores(self, model=None, features_name = None):
        
        
        if model is None:   
            model = self.oplsda
        else:
            model = model
            
        self.features_name = features_name
        
        features_name = self.features_name
        model = self.model


        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
            vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
       
        if features_name is not None:
            vips = pd.DataFrame(vips, columns = ['VIP'])
            vips['Features'] = features_name
        else:
            vips = pd.DataFrame(vips, columns = ['VIP'])
            vips['Features'] = vips.index

            
        self.vips = vips

        return

    def get_vip_scores(self):
        vips = self.vips
        return vips



    def vip_plot(self, threshold = 2):
        
        
        # add scatter plot of VIP score
        import plotly.express as px
        vips = self.vips

        fig = px.scatter(vips, x='Features', y='VIP', text='Features', color='VIP', color_continuous_scale='jet', range_color=(0, 2.5), height=500, width=1000, title='VIP score')
        fig.update_traces(marker=dict(size=12))
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        fig.update_yaxes(tickformat=",.00")
        fig.update_xaxes(tickformat=",.00")
        fig.update_layout(
            title={
                'y':1,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            font=dict(size=20))

        # reverse the x-axis
        fig.update_xaxes(autorange="reversed")
        
        # add dashed line for threshold
        fig.add_shape(type="line",
                    x0=0, y0=threshold, x1=10, y1=threshold,
                    line=dict(color="red",width=2, dash="dash"))
                    
        fig.update_layout(showlegend=False)
        
        return fig

