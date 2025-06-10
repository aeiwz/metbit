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


class nmr_preprocessing:
    '''
    A class for preprocessing NMR data.
    This class handles the following preprocessing steps:
    1. Reading FID files
    2. Zero-filling
    3. Fourier Transform
    4. Phasing
    5. Baseline correction
    6. Calibration
    7. Data storage in a pandas DataFrame
    8. Data visualization
    9. Data export
    
    Parameters:
    ----------
    data_path : str
        Path to the directory containing FID files.
    bin_size : float
        Size of the bins for zero-filling (default: 0.0003).
    auto_phasing : bool
        If True, automatic phasing is applied (default: True).
    fn_ : str
        Function name for phasing (default: 'acme').
    baseline_correction : bool
        If True, baseline correction is applied (default: True).
    baseline_type : str
        Type of baseline correction to apply (default: 'linear').
        Options: 'corrector', 'constant', 'explicit', 'median', 'solvent filter'.
    calibration : bool
        If True, calibration is applied (default: True).
    calib_type : str
        Type of calibration to apply (default: 'tsp').
        Options: 'tsp', 'acetate', 'glucose', 'alanine', 'formate'.
    custom_range : tuple
        Optional (start, end) PPM range for custom calibration.
        export_path : str
        Path to save the processed data (default: None).
        export_format : str
        Format to save the processed data (default: 'csv').
        export_name : str
        Name of the exported file (default: 'processed_nmr_data').
        
    Attributes:
    ----------
    nmr_data : pd.DataFrame
        Processed NMR data.
    ppm : np.ndarray
        PPM scale.
    dic_array : dict    
        Dictionary containing metadata from the FID files.
    phase_data : pd.DataFrame
        DataFrame containing phase information.
    
    Methods:
    -------
    get_data() : pd.DataFrame
        Returns the processed NMR data.
    get_ppm() : np.ndarray
        Returns the PPM scale.
    get_metadata() : dict
        Returns the metadata from the FID files.
    get_phase() : pd.DataFrame
        Returns the phase information.
    plot_data() : None
        Plots the processed NMR data.
    export_data() : None
        Exports the processed NMR data to a specified format.
    
    Example:
    -------
    >>> fid = 'dev/launch/data/test_nmr_data'
    >>> nmr = nmr_preprocessing(fid, bin_size=0.0005, auto_phasing=False, fn_='acme',
                                baseline_correction=True, baseline_type='corrector',
                                calibration=True, calib_type='glucose')
    >>> data = nmr.get_data()
    >>> ppm = nmr.get_ppm()
    >>> metadata = nmr.get_metadata()
    
    '''
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
                print(f"[DEBUG] Manual phasing with p0={dic['procs']['PHC0']} and p1={dic['procs']['PHC1']}")
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
        nmr_data.columns = ppm[::-1]
        try:
            nmr_data.index = dir_['folder name'].astype(int).to_list()
        except:
            nmr_data.index = dir_['folder name'].to_list()
        phase_data.index = nmr_data.index


        from .calibrate import calibrate
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
    fid = 'dev/launch/data/test_nmr_data'
    nmr = nmr_preprocessing(fid, bin_size=0.0005, auto_phasing=False, fn_='acme',
                            baseline_correction=True, baseline_type='corrector',
                            calibration=True, calib_type='glucose')
    #print(nmr.get_data().shape())
    #print(nmr.get_data().head())
    
    nmr_data = nmr.get_data()
    print(nmr_data.columns)
    #plot data
    from lingress import plot_NMR_spec
    fig = plot_NMR_spec(spectra=nmr_data, ppm=nmr.get_ppm(), label=None).single_spectra()
    fig.show()