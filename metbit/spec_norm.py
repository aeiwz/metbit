# -*- coding: utf-8 -*-

__author__ = 'aeiwz'
__copyright__="Copyright 2024, Theerayut"

__license__ = "MIT"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Development"

import numpy as np
import pandas as pd
from typing import Union


class Normalization:
    """
    A collection of lightweight normalization utilities (PQN, SNV, MSC and their combinations).
    Methods accept either pandas DataFrames or array-like inputs and always return DataFrames.
    """

    @staticmethod
    def _to_dataframe(df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Coerce input to DataFrame while preserving columns when possible."""
        if isinstance(df, pd.DataFrame):
            return df.copy()
        try:
            return pd.DataFrame(df)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError("Input data is not a valid DataFrame or convertible array.") from exc

    @staticmethod
    def pqn_normalization(df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Probabilistic Quotient Normalization (PQN)."""
        df = Normalization._to_dataframe(df)
        median_spectra = df.median(axis=0)
        safe_median = median_spectra.replace(0, np.nan)
        fold_change = df.divide(safe_median, axis=1)
        pqn_coef = fold_change.median(axis=1).replace(0, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            norm_df = df.divide(pqn_coef, axis=0)
        return norm_df.fillna(0)

    @staticmethod
    def snv_normalization(df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Standard Normal Variate (column-wise mean centering and scaling)."""
        df = Normalization._to_dataframe(df)
        mean = df.mean(axis=0)
        std = df.std(axis=0, ddof=0).replace(0, np.nan)
        return (df - mean).div(std, axis=1)

    @staticmethod
    def msc_normalization(df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Multiplicative Scatter Correction."""
        df = Normalization._to_dataframe(df)
        centered = df - df.mean(axis=0)
        scaled = centered.div(centered.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
        return scaled - scaled.mean(axis=0)

    @staticmethod
    def snv_msc_normalization(df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Apply SNV followed by MSC-style column centering."""
        snv = Normalization.snv_normalization(df)
        return snv - snv.mean(axis=0)

    @staticmethod
    def snv_pqn_normalization(df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Apply SNV followed by PQN normalization."""
        df = Normalization._to_dataframe(df)
        median = df.median(axis=1).replace(0, np.nan)
        snv = df.div(median, axis=0)
        return snv.sub(snv.mean(axis=1), axis=0)

    @staticmethod
    def snv_msc_pqn_normalization(df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Apply SNV, MSC-style centering, then PQN."""
        df = Normalization._to_dataframe(df)
        median = df.median(axis=1).replace(0, np.nan)
        snv = df.div(median, axis=0)
        centered_rows = snv.sub(snv.mean(axis=1), axis=0)
        return centered_rows - centered_rows.mean(axis=0)

   
