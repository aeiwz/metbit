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
    from scipy.signal import savgol_filter
    import pandas as pd
    import numpy as np

    print(f"[DEBUG] Starting calibration: type={calib_type}, custom_range={custom_range}")

    if not isinstance(X, pd.DataFrame):
        print("[DEBUG] Converting X to DataFrame")
        X = pd.DataFrame(X)
        X.columns = ppm

    calib_ranges = {
        'tsp': (-0.2, 0.2),
        'glucose': (5.15, 5.3),
        'alanine': (1.4, 1.56)
    }

    if calib_type in calib_ranges:
        ppm_range = calib_ranges[calib_type]
        target_ppm = {
            'tsp': 0.0,
            'glucose': 5.23,
            'alanine': 1.48
        }[calib_type]
    elif custom_range:
        ppm_range = custom_range
        target_ppm = np.mean(custom_range)
    else:
        raise ValueError("Invalid calibration type or custom range.")

    print(f"[DEBUG] Calibration range: {ppm_range}, target: {target_ppm}")

    range_mask = (ppm >= ppm_range[0]) & (ppm <= ppm_range[1])
    if not np.any(range_mask):
        raise ValueError(f"No ppm values found within range {ppm_range}.")

    calibrated_X = np.zeros_like(X.values)
    for i, spectrum in enumerate(X.values):
        segment = spectrum[range_mask]
        segment_ppm = ppm[range_mask]
        peak_idx = np.argmax(segment)
        peak_ppm = segment_ppm[peak_idx]
        shift = peak_ppm - target_ppm
        adjusted_ppm = ppm - shift
        calibrated_X[i, :] = np.interp(ppm, adjusted_ppm, spectrum)
        print(f"[DEBUG] Sample {i}: peak={peak_ppm}, shift={shift}")

    calibrated_X_df = pd.DataFrame(calibrated_X)
    calibrated_X_df.columns = ppm
    calibrated_X_df.index = X.index

    print("[DEBUG] Calibration completed.")
    return calibrated_X_df

class nmr_preprocessing:
    def __init__(self, data_path: str, bin_size: float = 0.0003, 
                auto_phasing: bool = True, fn_ = 'acme',
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
                pass

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
            self.nmr_data = calibrate(nmr_data, ppm, calib_type)



        print("\n[INFO] Data processing completed\n")
        self.nmr_data = nmr_data
        self.ppm = ppm
        self.dic_array = dic_list
        self.phase_data = phase_data
        print("[DEBUG] Type of nmr_data:", type(self.nmr_data))



    def get_data(self):
        print("[DEBUG] get_data() called")
        if self.nmr_data is None or not isinstance(self.nmr_data, pd.DataFrame):
            raise ValueError("NMR data is not available or not in DataFrame format.")
        if self.nmr_data.empty:
            raise ValueError("NMR data is empty.")
        if self.nmr_data.isnull().values.any():
            raise ValueError("NMR data contains missing values.")
        # Check if the index is numeric
        if not pd.api.types.is_numeric_dtype(self.nmr_data.index):
            raise ValueError("NMR data index is not numeric.")
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
