# -*- coding: utf-8 -*-

__author__ = 'aeiwz'
__copyright__="Copyright 2025, Theerayut"

__license__ = "MIT"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Development"



class AutoPipe:

    '''
    Documentation:
    -------------
    
    Parameters:
    -----------


    Returns:
    --------


    Example:
    --------

    
    '''

    import pandas as pd

    def __init__(self, data: pd.DataFrame, target_col: str, feature_names: list, working_dir: str,):
        self.data = data
        self.target_col = target_col
        self.feature_names = feature_names
        self.working_dir = working_dir

    def analysis(self):
        pass
