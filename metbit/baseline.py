def bline(X: pd.DataFrame, lam: float = 1e7, max_iter: int = 30) -> pd.DataFrame:
    """
    Baseline correction for 1D NMR spectra using asymmetric least squares (ALS).
    
    Parameters:
        X (pd.DataFrame): DataFrame where rows are spectra, columns are PPM values.
        lam (float): Smoothing parameter (lambda). Higher = smoother baseline.
        max_iter (int): Max iterations for ALS.
    
    Returns:
        pd.DataFrame: Baseline-corrected spectra (same shape as input).
    """
    if X.isnull().values.any():
        print("[WARNING] Data contains missing values. Replacing with zeros.")
        X = X.fillna(0)

    corrected = []
    for idx, spectrum in X.iterrows():
        baseline, _ = asls(spectrum.values, lam=lam, max_iter=max_iter)
        corrected.append(spectrum.values - baseline)

    corrected_df = pd.DataFrame(corrected, index=X.index, columns=X.columns)
    return corrected_df