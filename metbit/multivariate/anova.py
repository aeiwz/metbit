# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


__author__ = "aeiwz"

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import f


class anova_oplsda:
    '''
    ANOVA for OPLS-DA model
    
    This class implements ANOVA analysis for the OPLS-DA (Orthogonal Partial Least Squares Discriminant Analysis) model.
    It calculates the F-statistics and p-values for each number of components in the OPLS-DA model.
    
    Parameters:
    - X: predictor variables as numpy array or pandas DataFrame
    - Y: response variable as numpy array or pandas Series (categorical)
    - n_components: number of components for the OPLS-DA model (default: 2)
    - cv: number of folds for cross-validation (default: 5)
    
    Methods:
    - fit(): Fits the OPLS-DA model and calculates the F-statistics and p-values.
    - summary(): Generates a summary table with the F-statistics and p-values for each number of components.
    '''
    def __init__(self, X, Y, n_components=2, cv=5):
        self.X = X
        self.Y = Y
        self.n_components = n_components
        self.cv = cv
        
        parameter = f'''
        X: predictor variables as {type(X)} of shape {X.shape} \n
        Y: response variable as {type(Y)} of shape {Y.shape} \n
        n_components: {n_components} components for OLSDA model \n
        cv: {cv}-fold \n
        '''
        print(parameter)
        
   
    def fit(self):
        '''
        Fits the OPLS-DA model and calculates the F-statistics and p-values.
        '''
        # X: predictor variables (numpy array or pandas DataFrame)
        # Y: response variable (categorical) (numpy array or pandas Series)
        # n_components: number of components in the OPLS-DA model
        X = self.X
        Y = self.Y
        cv = self.cv
        n_components = self.n_components
        # Convert Y to numeric labels
        classes, Y_numeric = np.unique(Y, return_inverse=True)
        
        # Initialize lists to store ANOVA results
        mse_between = []
        mse_within = []
        
        for i in range(n_components):
            # Fit OPLS-DA model with i+1 components
            oplsda = PLSRegression(n_components=i+1)
            y_pred = cross_val_predict(oplsda, X, Y_numeric, cv=5)
            
            # Calculate mean squared error between groups and within groups
            mse_between.append(np.mean((np.mean(y_pred, axis=0) - np.mean(Y_numeric, axis=0))**2))
            mse_within.append(np.mean(np.var(y_pred, axis=0)))
        
        # Calculate F-statistics and p-values
        f_statistics = np.array(mse_between) / np.array(mse_within)
        p_values = 1 - f.cdf(f_statistics, n_components-1, len(Y)-n_components)
        
        self.f_statistics = f_statistics
        self.p_values = p_values
        
    
    
    def summary(self):
        '''
        Generates a summary table with the F-statistics and p-values for each number of components.
        
        Returns:
        - summary_table: pandas DataFrame with the F-statistics and p-values
        '''
        # Create summary table
        f_statistics = self.f_statistics
        p_values = self.p_values
        summary_table = pd.DataFrame({'F-statistic': f_statistics, 'p-value': p_values})
        summary_table.index.name = 'Number of components'
        
        self.summary_table = summary_table
        return summary_table
    
    
    
