# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

__author__ = "aeiwz"

class unipair:

    def __init__(self, dataset, column_name):
        
        meta = dataset
        self.meta = meta
        self.column_name = column_name
        

        """
        This function takes in a dataframe and a column name and returns the index of the dataframe and the names of the pairs
        of the unique values in the column.
        Parameters
        ----------
        meta: pandas dataframe
            The dataframe to be used.
        column_name: str
        Unipair(meta, column_name).indexing()
        
        """
        import pandas as pd
        import numpy as np
        
        #check unique values in the column
        if meta[column_name].nunique() < 3:
            raise ValueError("Group should contain at least 3 groups")
        else:
            pass
        #check meta is a dataframe
        if not isinstance(meta, pd.DataFrame):
            raise ValueError("meta should be a pandas dataframe")
        #check column_name is a string
        if not isinstance(column_name, str):
            raise ValueError("column_name should be a string")
        

        df = meta
        y = df[column_name].unique()
        pairs = []
        for i in range(len(y)):
            for j in range(i+1, len(y)):
                pairs.append([y[i], y[j]])
        
        index_ = []
        for i in range(len(pairs)):
            inside_index = []
            for j in range(2):
                inside_index.append(list((df.loc[df[column_name] == pairs[i][j]]).index))
            index_list = [inside_index[0] + inside_index[1]]
            index_.append(index_list[0])
        pairs
        index_
        names = []
        for i in range(len(pairs)):
            
            names.append(str(pairs[i][0]) + "_vs_" + str(pairs[i][1]))
            #check names if contain / replace with _ 
            names[i] = names[i].replace('/', '_')
            
        del df
        del y
        
        self.index_ = index_
        self.names = names
        
        
        

    def get_index(self):
        index_ = self.index_
        return index_
    
    def get_name(self):
        names = self.names
        return names
    
    def get_meta(self):
        meta = self.meta
        column_name = self.column_name
        return meta[column_name]
    
    def get_column_name(self):
        column_name = self.column_name
        return column_name
    
    def get_dataset(self):
        df = self.meta
        index_ = self.index_
        list_of_df = []
        for i in range(len(index_)):
            list_of_df.append(df.loc[index_[i]])
        
        #Create object attribute
        self.list_of_df = list_of_df
        return list_of_df
        


