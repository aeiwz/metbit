def calibrate(X, ppm, calib_type='tsp', custom_range=None):
    """
    Calibrate chemical shifts in NMR spectra using known reference peaks.

    Parameters:
        X (np.ndarray or pd.DataFrame): NMR data matrix (rows: spectra, columns: PPM positions).
        ppm (np.ndarray): 1D array of chemical shift values.
        calib_type (str): 'tsp', 'glucose', 'alanine', or 'custom'.
        custom_range (tuple): Optional (start, end) PPM range for custom calibration.

    Returns:
        pd.DataFrame: Calibrated NMR data matrix.
    """
    import numpy as np
    import pandas as pd
    from scipy.signal import find_peaks

    # Define calibration ranges and targets
    calib_ranges = {
        'tsp': (-0.2, 0.2),
        'acetate': (1.8, 2.2),
        'glucose': (5.0, 5.4),
        'alanine': (1.2, 1.6),
        'formate': (8.0, 8.4),
    }
    target_ppm_dict = {
        'tsp': 0.000,
        'acetate': 1.910,
        'glucose': 5.230,
        'alanine': 1.480,
        'formate': 8.440,
    }

    # Determine calibration range and target ppm
    if calib_type in calib_ranges:
        ppm_range = calib_ranges[calib_type]
        target_ppm = target_ppm_dict[calib_type]
    elif custom_range:
        ppm_range = custom_range
        target_ppm = np.mean(custom_range)
    else:
        raise ValueError("Invalid calibration type or custom range.")

    # Ensure X is 2D NumPy array
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        X_index = X.index
        X = X.to_numpy()
    else:
        X_index = range(np.atleast_2d(X).shape[0])
        X = np.atleast_2d(X)

    # Check that ppm range is valid
    range_mask = (ppm >= ppm_range[0]) & (ppm <= ppm_range[1])
    if not np.any(range_mask):
        raise ValueError(f"No ppm values found within range {ppm_range}.")

    calibrated_X = np.zeros_like(X)

    for i, spectrum in enumerate(X):
        segment = spectrum[range_mask]
        segment_ppm = ppm[range_mask]

        if calib_type in ['glucose', 'alanine']:
            # Detect peaks in the segment
            peaks, _ = find_peaks(segment, prominence=0.01)
            if len(peaks) >= 2:
                # Get top 2 peaks (most intense)
                top2_idx = peaks[np.argsort(segment[peaks])[-2:]]
                top2_ppms = segment_ppm[top2_idx]
                peak_ppm = np.mean(top2_ppms)
            elif len(peaks) == 1:
                peak_ppm = segment_ppm[peaks[0]]
            else:
                peak_ppm = segment_ppm[np.argmax(segment)]
        else:
            # Use max intensity in region
            peak_idx = np.argmax(segment)
            peak_ppm = segment_ppm[peak_idx]

        shift = peak_ppm - target_ppm
        adjusted_ppm = ppm - shift

        # Sort before interpolation to avoid np.interp issues
        sort_idx = np.argsort(adjusted_ppm)
        calibrated_X[i, :] = np.interp(ppm, adjusted_ppm[sort_idx], spectrum[sort_idx])

    # Return as DataFrame
    return pd.DataFrame(calibrated_X, columns=ppm, index=X_index)