
#Preprocessing data set to decrease noise
class Denoise:
    def decrease_noise(spectra, window_length=11, polyorder=2):
        import pandas as pd
        import numpy as np
        from scipy.signal import savgol_filter
        """
        Decrease the noise of spectra using Savitzky-Golay filter.
        
        Parameters:
        - spectra: numpy array or pandas DataFrame
            The spectra data to be processed.
        - window_length: int, optional (default=11)
            The length of the window used for filtering.
        - polyorder: int, optional (default=2)
            The order of the polynomial used for fitting.
        
        Returns:
        - filtered_spectra: numpy array or pandas DataFrame
            The spectra data after noise reduction.
        """
        if isinstance(spectra, np.ndarray):
            filtered_spectra = savgol_filter(spectra, window_length, polyorder, axis=1)
        elif isinstance(spectra, pd.DataFrame):
            filtered_spectra = spectra.apply(lambda x: savgol_filter(x, window_length, polyorder))
        else:
            raise ValueError("Invalid data type. Expected numpy array or pandas DataFrame.")
        
        return filtered_spectra
