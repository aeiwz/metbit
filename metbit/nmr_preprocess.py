# -*- coding: utf-8 -*-

__author__ = 'aeiwz'
author_email='theerayut_aeiw_123@hotmail.com'
__copyright__="Copyright 2024, Theerayut"
__license__ = "MIT"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Develop"

def read_fid(data_path: str):
    import nmrglue as ng
    print(f"[DEBUG] Reading FID from: {data_path}")
    dic, data = ng.bruker.read(data_path)
    return dic, data

def remove_digital_filter(dic, data):
    import nmrglue as ng
    print("[DEBUG] Removing digital filter")
    data = ng.bruker.remove_digital_filter(dic, data)
    return data

def generate_ppm_scale(dic, data):
    import numpy as np
    size = len(data)
    print(f"[DEBUG] Data size: {size}")
    sweep_width = dic['acqus']['SW']
    spectrometer_freq = dic['acqus']['SFO1']
    offset = dic['procs']['OFFSET']
    print(f"[DEBUG] SW={sweep_width}, SFO1={spectrometer_freq}, OFFSET={offset}")
    ppm = np.linspace(offset, offset - sweep_width, size)
    return ppm

def phasing(data, index, auto=True, fn='peak_minima', p0=0.0, p1=0.0):
    import nmrglue as ng
    print(f"[DEBUG] Phasing spectrum at index {index}, auto={auto}, fn={fn}")
    if auto:
        data[index] = ng.process.proc_autophase.autops(data[index], fn=fn, p0=p0, p1=p1, return_phases=False)
    return data

import numpy as np
import pandas as pd
from pybaselines.whittaker import asls

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
        'alanine': (1.2, 1.6)
    }
    target_ppm_dict = {
        'tsp': 0.000,
        'acetate': 1.910,
        'glucose': 5.230,
        'alanine': 1.480
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

class nmr_preprocessing:
    def __init__(self, data_path: str, bin_size: float = 0.0003, 
                auto_phasing: bool = False, fn_ = 'acme',
                baseline_correction: bool = True, baseline_type: str = 'linear', 
                calibration: bool = True, calib_type: str = 'tsp'):

        print("[DEBUG] Initializing nmr_preprocessing class")
        self.data_path = data_path
        self.bin_size = bin_size
        self.auto_phasing = auto_phasing
        self.baseline_correction = baseline_correction
        self.calibration = calibration
        self.calib_type = calib_type
        self.baseline_type = baseline_type
        self.fn_ = fn_

        import os
        from glob import glob
        import tqdm
        import nmrglue as ng
        import pandas as pd
        import numpy as np

        print(f"[DEBUG] Checking path: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path '{data_path}' not found.")
        if data_path.endswith('/'):
            data_path = data_path[:-1]
        
        # Search for fid directories at different depths
        sub_dirs = glob(f'{data_path}/fid')
        if not sub_dirs:
            sub_dirs = glob(f'{data_path}/*/fid')
        if not sub_dirs:
            sub_dirs = glob(f'{data_path}/*/*/fid')

        # Fail if still nothing found
        if not sub_dirs:
            raise ValueError(f"No 'fid' files found in '{data_path}'. Please check your folder structure.")

        print(f"[DEBUG] Found {len(sub_dirs)} fid directories:")
        for p in sub_dirs:
            print("  └", p)

        dir_ = pd.DataFrame(sub_dirs, columns=['dir'])
        dir_['dir'] = dir_['dir'].apply(lambda x: x.replace('\\', '/'))
        dir_['folder name'] = dir_['dir'].apply(lambda x: x.split('/')[-2])
        dir_['dir'].replace('fid', '', regex=True, inplace=True)

        nmr_data = pd.DataFrame()
        phase_data = pd.DataFrame(columns=['p0', 'p1'])
        dic_list = {}
        ppm = None

        #sort index of dir_ by folder name
        dir_.sort_values('folder name', inplace=True)
        for i in tqdm.tqdm(dir_.index):
            path_to_process = dir_['dir'][i]
            print(f"[DEBUG] Processing sample {i}: {path_to_process}")

            try:
                dic, data = ng.bruker.read(path_to_process)
            except Exception as e:
                print(f"[ERROR] Reading Bruker data failed: {e}")
                raise

            print("[DEBUG] Removing digital filter")
            data = ng.bruker.remove_digital_filter(dic, data)
            size = len(data)
            sweep_width = dic['acqus']['SW']
            spectrometer_freq = dic['acqus']['SFO1']
            offset = dic['procs']['OFFSET']

            print(f"[DEBUG] ZF size calc: sweep_width={sweep_width}, bin_size={bin_size}")
            zf_size = int(sweep_width / bin_size)
            data = ng.proc_base.zf_size(data, zf_size)
            print(f"[DEBUG] FFT")
            data = ng.proc_base.fft(data)

            if auto_phasing:
                print("[DEBUG] Auto phasing")
                data, (p0, p1) = ng.process.proc_autophase.autops(data, fn=fn_, return_phases=True)
                phase = [p0, p1]
                print(f"[DEBUG] Phase angles: p0={p0}, p1={p1}")
            else:
                print("[DEBUG] Manual phasing")
                data = ng.process.proc_base.ps(data, p0=dic['procs']['PHC0'], p1=dic['procs']['PHC1'], inv=True)
                phase = [dic['procs']['PHC0'], dic['procs']['PHC1']]

            data = ng.proc_base.di(data)
            data = ng.proc_base.rev(data)



            # Perform baseline correction
            if baseline_correction:
                if baseline_type == 'corrector':
                    data = ng.process.proc_bl.baseline_corrector(data)
                elif baseline_type == 'constant':
                    data = ng.process.proc_bl.cbf(data)
                elif baseline_type == 'explicit':
                    data = ng.process.proc_bl.cbf_explicit(data)
                elif baseline_type == 'median':
                    data = ng.process.proc_bl.med(data)
                elif baseline_type == 'solvent filter':
                    data = ng.process.proc_bl.sol_gaussian(data)
                else:
                    pass
            else:
                pass

            ppm = np.linspace(offset, offset - sweep_width, zf_size)
            print(f"[DEBUG] Generated PPM scale (length={len(ppm)})")

            nmr_data = pd.concat([nmr_data, pd.DataFrame([data.real])], axis=0)
            dic_list.update({dir_['folder name'][i]: dic})
            phase_data = pd.concat([phase_data, pd.DataFrame(phase).T], axis=0)

        '''
        if self.baseline_correction:
            print("[DEBUG] Starting baseline correction")
            nmr_data = bline(nmr_data, lam=1e7, max_iter=30)
        '''

        print("[DEBUG] Setting column names and index")
        nmr_data.columns = ppm
        try:
            nmr_data.index = dir_['folder name'].astype(int).to_list()
        except:
            nmr_data.index = dir_['folder name'].to_list()
        phase_data.index = nmr_data.index


        if self.calibration:
            print("[DEBUG] Starting calibration step")
            self.nmr_data2 = calibrate(nmr_data, ppm, calib_type)

        print("\n[INFO] Data processing completed\n")
        if self.calibration:
            print("[DEBUG] Calibration completed")
            self.nmr_data = self.nmr_data2
        else:
            print("[DEBUG] No calibration applied")
            self.nmr_data = nmr_data
        #self.nmr_data = nmr_data
        self.ppm = ppm
        self.dic_array = dic_list
        self.phase_data = phase_data
        print("[DEBUG] Type of nmr_data:", type(self.nmr_data))


    def get_data(self):
        print("[DEBUG] get_data() called")
        return self.nmr_data

    def get_ppm(self):
        print("[DEBUG] get_ppm() called")
        return self.ppm

    def get_metadata(self):
        print("[DEBUG] get_metadata() called")
        return self.dic_array

    def get_phase(self):
        print("[DEBUG] get_phase() called")
        return self.phase_data



if __name__ == '__main__':
    print("[DEBUG] Starting NMR processing script")
    fid = '/Volumes/CAS9/Aeiwz/Project/Thesis/Archive'
    nmr = nmr_preprocessing(fid, bin_size=0.0003, auto_phasing=True, fn_='acme',
                            baseline_correction=True, baseline_type='linear',
                            calibration=True, calib_type='glucose')
    #print(nmr.get_data().shape())
    #print(nmr.get_data().head())
    
    nmr_data = nmr.get_data()
    #plot data
    from lingress import plot_NMR_spec
    fig = plot_NMR_spec(spectra=nmr_data, ppm=nmr.get_ppm(), label=None).single_spectra()
    fig.show()
