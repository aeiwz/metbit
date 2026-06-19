# -*- coding: utf-8 -*-

import pandas as pd


class lazypair:
    """Utility for generating all pairwise groupings and related dataset splits."""

    def __init__(self, dataset: pd.DataFrame, column_name: str):
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("meta should be a pandas dataframe")
        if not isinstance(column_name, str):
            raise ValueError("column_name should be a string")
        if column_name not in dataset.columns:
            raise KeyError(f"{column_name} not found in dataset")
        if dataset[column_name].nunique() < 2:
            raise ValueError("Group should contain at least 2 groups")

        self.meta = dataset
        self.column_name = column_name

        groups = dataset[column_name].unique()
        self.pairs = [(g1, g2) for i, g1 in enumerate(groups) for g2 in groups[i + 1:]]
        self.index_ = [
            dataset.index[dataset[column_name] == g1].tolist()
            + dataset.index[dataset[column_name] == g2].tolist()
            for g1, g2 in self.pairs
        ]
        self.names = [f"{g1}_vs_{g2}".replace("/", "_") for g1, g2 in self.pairs]

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
