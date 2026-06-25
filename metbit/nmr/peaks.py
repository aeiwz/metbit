# -*- coding: utf-8 -*-

__author__ = 'aeiwz'
__copyright__="Copyright 2024, Theerayut"

__license__ = "MIT"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Development"

import pandas as pd
import numpy as np

class peak_chops:
    """Utility class for removing (chopping) unwanted spectral regions from NMR data.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> ppm = np.linspace(10, 0, 1000)
        >>> spectra = pd.DataFrame(np.random.rand(10, 1000), columns=ppm)
        >>> chopper = metbit.nmr.peaks.peak_chops(spectra)
        >>> X_chopped, ppm_chopped = chopper.cut_peak(4.7, 5.0)
    """

    def __init__(self, data: pd.DataFrame, ppm: list = None) -> None:
        '''
        data: NMR spectra data maybe contain ppm as columns
        ppm: list of ppm (Can ignore this argument if columns of data is ppm)

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> ppm = np.linspace(10, 0, 1000)
            >>> spectra = pd.DataFrame(np.random.rand(10, 1000), columns=ppm)
            >>> chopper = metbit.nmr.peaks.peak_chops(spectra)
            >>> chopper2 = metbit.nmr.peaks.peak_chops(spectra, ppm=list(ppm))
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
                ppm = data.columns.astype(float)
            except Exception:
                raise ValueError(text)

        else:
            pass

        if ppm is not None:
            if ppm is not pd.core.indexes.base.Index:
                ppm = pd.core.indexes.base.Index(ppm)
            else:
                pass  # pragma: no cover

        self.data = data
        self.ppm = ppm




    def cut_peak(self, first_ppm: float, second_ppm: float):
        """Remove a spectral region between two ppm values.

        Args:
            first_ppm: Start of the region to remove (ppm).
            second_ppm: End of the region to remove (ppm).

        Returns:
            tuple: (X, ppm) where X is the trimmed DataFrame and ppm is the
                updated list of chemical shift values.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> ppm = np.linspace(10, 0, 1000)
            >>> spectra = pd.DataFrame(np.random.rand(10, 1000), columns=ppm)
            >>> chopper = metbit.nmr.peaks.peak_chops(spectra)
            >>> X_chopped, ppm_chopped = chopper.cut_peak(4.7, 5.0)
            >>> print(X_chopped.shape)
        """

        X = self.data
        ppm = self.ppm


        X.columns = ppm

        first_index = ppm.get_loc(min(ppm, key=lambda x: abs(x - first_ppm)))
        second_index = ppm.get_loc(min(ppm, key=lambda x: abs(x - second_ppm))) + 1

        X.drop(columns=X.iloc[:, first_index:second_index].columns, inplace=True)
        ppm = X.columns.astype(float).to_list()

        return X, ppm
