# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np


class peak_chops:

    def __init__(self, data: pd.DataFrame, ppm: list = None) -> None:

        '''
        data: NMR spectra data maybe contain ppm as columns
        ppm: list of ppm (Can ignore this argument if columns of data is ppm)
        '''

        #if ppm is None check column of X can convert to float?
        if ppm is None:
            text = f'''
            ------------------------------------------- \n
            The columns of data can't convert to ppm \n
            please assign ppm parameter as list of ppm \n
            -------------------------------------------
            '''
            try:
                ppm = data.columns.astype(float).to_list()
            except:
                raise print(text)

        else:
            pass

        self.data = data
        self.ppm = ppm




    def cut_peak(self, first_ppm: float, second_ppm: float):
        
        X = self.data
        ppm = self.ppm 


        
        first_index = ppm.get_loc(min(ppm, key=lambda x: abs(x - first_ppm)))
        second_index = ppm.get_loc(min(ppm, key=lambda x: abs(x - second_ppm)))
        
        X.drop(columns=X.iloc[:, first_index:second_index].columns, inplace=True)
        ppm = X.columns.astype(float)
        
        return X, ppm