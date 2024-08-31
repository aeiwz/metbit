# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np


class peak_chops:

    def __init__(self, data: pd.DataFrame, first_region: float, second_region: float):
        self.data = data
        self.first_region = first_region
        self.second_region = second_region

    def cut_peak(X: pd.DataFrame, first_ppm: float, second_ppm: float):
    
        import pandas as pd
        
        ppm = X.columns.astype(float)
        
        first_index = ppm.get_loc(min(ppm, key=lambda x: abs(x - first_ppm)))
        second_index = ppm.get_loc(min(ppm, key=lambda x: abs(x - second_ppm)))
        
        X.drop(columns=X.iloc[:, first_index:second_index].columns, inplace=True)
        ppm = X.columns.astype(float)
        
        return X, ppm