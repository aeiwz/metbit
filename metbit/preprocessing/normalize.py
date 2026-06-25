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

from metbit._native import pqn_median_quotient as _pqn_median


class Normalization:
    """
    A collection of lightweight normalization utilities (PQN, SNV, MSC and their combinations).
    Methods accept either pandas DataFrames or array-like inputs and always return DataFrames.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.preprocessing.normalize import Normalization
        >>> spectra = pd.DataFrame(np.random.rand(10, 100))
        >>> norm_spectra = Normalization.pqn_normalization(spectra)
        >>> norm_spectra.shape
        (10, 100)
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
        """Probabilistic Quotient Normalization (PQN).

        Args:
            df: Spectral data with samples as rows and variables as columns.

        Returns:
            PQN-normalized DataFrame of the same shape.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.preprocessing.normalize import Normalization
            >>> spectra = pd.DataFrame(np.random.rand(10, 50))
            >>> norm_spectra = Normalization.pqn_normalization(spectra)
            >>> norm_spectra.shape
            (10, 50)
        """
        df = Normalization._to_dataframe(df)
        X = df.values.astype(np.float64)
        reference = np.median(X, axis=0)
        # per-sample median quotient via native kernel
        coefs = np.array([_pqn_median(X[i], reference) for i in range(X.shape[0])])
        coefs = np.where(coefs == 0.0, np.nan, coefs)
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = X / coefs[:, np.newaxis]
        return pd.DataFrame(np.nan_to_num(norm, nan=0.0),
                            index=df.index, columns=df.columns)

    @staticmethod
    def snv_normalization(df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Standard Normal Variate (column-wise mean centering and scaling).

        Args:
            df: Spectral data with samples as rows and variables as columns.

        Returns:
            SNV-normalized DataFrame of the same shape.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.preprocessing.normalize import Normalization
            >>> spectra = pd.DataFrame(np.random.rand(10, 50))
            >>> norm_spectra = Normalization.snv_normalization(spectra)
            >>> norm_spectra.shape
            (10, 50)
        """
        df = Normalization._to_dataframe(df)
        mean = df.mean(axis=0)
        std = df.std(axis=0, ddof=0).replace(0, np.nan)
        return (df - mean).div(std, axis=1)

    @staticmethod
    def msc_normalization(df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Multiplicative Scatter Correction.

        Args:
            df: Spectral data with samples as rows and variables as columns.

        Returns:
            MSC-normalized DataFrame of the same shape.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.preprocessing.normalize import Normalization
            >>> spectra = pd.DataFrame(np.random.rand(10, 50))
            >>> norm_spectra = Normalization.msc_normalization(spectra)
            >>> norm_spectra.shape
            (10, 50)
        """
        df = Normalization._to_dataframe(df)
        centered = df - df.mean(axis=0)
        scaled = centered.div(centered.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
        return scaled - scaled.mean(axis=0)

    @staticmethod
    def snv_msc_normalization(df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Apply SNV followed by MSC-style column centering.

        Args:
            df: Spectral data with samples as rows and variables as columns.

        Returns:
            SNV+MSC-normalized DataFrame of the same shape.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.preprocessing.normalize import Normalization
            >>> spectra = pd.DataFrame(np.random.rand(10, 50))
            >>> norm_spectra = Normalization.snv_msc_normalization(spectra)
            >>> norm_spectra.shape
            (10, 50)
        """
        snv = Normalization.snv_normalization(df)
        return snv - snv.mean(axis=0)

    @staticmethod
    def snv_pqn_normalization(df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Apply SNV followed by PQN normalization.

        Args:
            df: Spectral data with samples as rows and variables as columns.

        Returns:
            SNV+PQN-normalized DataFrame of the same shape.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.preprocessing.normalize import Normalization
            >>> spectra = pd.DataFrame(np.random.rand(10, 50))
            >>> norm_spectra = Normalization.snv_pqn_normalization(spectra)
            >>> norm_spectra.shape
            (10, 50)
        """
        df = Normalization._to_dataframe(df)
        median = df.median(axis=1).replace(0, np.nan)
        snv = df.div(median, axis=0)
        return snv.sub(snv.mean(axis=1), axis=0)

    @staticmethod
    def snv_msc_pqn_normalization(df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Apply SNV, MSC-style centering, then PQN.

        Args:
            df: Spectral data with samples as rows and variables as columns.

        Returns:
            SNV+MSC+PQN-normalized DataFrame of the same shape.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from metbit.preprocessing.normalize import Normalization
            >>> spectra = pd.DataFrame(np.random.rand(10, 50))
            >>> norm_spectra = Normalization.snv_msc_pqn_normalization(spectra)
            >>> norm_spectra.shape
            (10, 50)
        """
        df = Normalization._to_dataframe(df)
        median = df.median(axis=1).replace(0, np.nan)
        snv = df.div(median, axis=0)
        centered_rows = snv.sub(snv.mean(axis=1), axis=0)
        return centered_rows - centered_rows.mean(axis=0)


