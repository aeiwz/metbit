# -*- coding: utf-8 -*-

__auther__ ='aeiwz'
author_email='theerayut_aeiw_123@hotmail.com'
__copyright__="Copyright 2024, Theerayut"

__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Develop"




def read_fid(data_path: str):
    """
    Read in NMR data from a Bruker FID file.
    
    Parameters:
        data_path (str): Path to the Bruker FID file.
    
    Returns:
        tuple: Tuple containing the data dictionary and the data array.
    """
    import nmrglue as ng
    
    # Read in the Bruker formatted data
    dic, data = ng.bruker.read(data_path)
    
    return dic, data

def remove_digital_filter(dic, data):
    """
    Remove the digital filter from NMR data.
    
    Parameters:
        dic (dict): Data dictionary.
        data (np.ndarray): NMR data array.
    
    Returns:
        np.ndarray: NMR data array with digital filter removed.
    """
    import nmrglue as ng
    
    # Remove the digital filter
    data = ng.bruker.remove_digital_filter(dic, data)
    
    return data

def generate_ppm_scale(dic, data):
    """
    Generate a PPM scale for NMR data.
    
    Parameters:
        dic (dict): Data dictionary.
        bin_size (float): Bin size for the PPM scale.
    
    Returns:
        np.ndarray: PPM scale array.
    """
    import numpy as np
    
    # Generate PPM scale manually
    size = len(data)
    sweep_width = dic['acqus']['SW']            # Spectral width in ppm
    spectrometer_freq = dic['acqus']['SFO1']    # Spectrometer frequency in MHz
    offset = dic['procs']['OFFSET']             # Offset in ppm

    # Generate PPM scale
    ppm = np.linspace(offset, offset - sweep_width, size)
    
    return ppm

def phasing(data, index, auto=True, fn='peak_minima', p0=0.0, p1=0.0):
    """
    Apply phase correction to NMR data.
    
    Parameters:
        data (np.ndarray): NMR data array.
        fn (str): Phase correction function ('peak_minima', 'peak_maxima', 'min_imag', 'max_imag').
        auto (bool): Perform automatic phase correction.
    
    Returns:
        np.ndarray: Phased NMR data array.
    """
    import nmrglue as ng
    
    # Perform phase correction
    if auto:
        data[index] = ng.process.proc_autophase.autops(data[index], fn='peak_minima', p0=0.0, p1=0.0, return_phases=False)

    return data




def calibrate(X, ppm, calib_type='tsp', custom_range=None):
    """
    Calibrate chemical shifts in NMR spectra.
    
    Parameters:
        X (np.ndarray): 2D array, NMR data (rows: spectra, columns: PPM positions).
        ppm (np.ndarray): 1D array, chemical shift values aligned with columns of X.
        calib_type (str): Calibration type ('tsp', 'glucose', 'alanine').
        custom_range (tuple): PPM range (start, end) for custom calibration.
    
    Returns:
        np.ndarray: Calibrated NMR data matrix.

    Example:
        # Ensure X is a 2D numpy array
        X = data.to_numpy()

        # Ensure ppm is a 1D numpy array
        ppm = data.columns[:-1].astype(float).to_numpy()

        # Call the calibrate function
        calibrated_X = calibrate(X=X, ppm=ppm, calib_type='tsp', custom_range=None)
    """    
    import numpy as np
    from scipy.signal import savgol_filter
    import pandas as pd
    # Define calibration ranges
    calib_ranges = {
        'tsp': (-0.2, 0.2),
        'glucose': (5.15, 5.3),
        'alanine': (1.4, 1.56)
    }

    # Determine calibration range
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

    # Find indices within the calibration range
    range_mask = (ppm >= ppm_range[0]) & (ppm <= ppm_range[1])
    if not np.any(range_mask):
        raise ValueError(f"No ppm values found within range {ppm_range}.")

    # Perform calibration
    calibrated_X = np.zeros_like(X)
    for i, spectrum in enumerate(X):
        # Extract the spectrum segment within the range
        segment = spectrum[range_mask]
        segment_ppm = ppm[range_mask]

        # Find the peak position
        peak_idx = np.argmax(segment)
        peak_ppm = segment_ppm[peak_idx]

        # Compute the shift to align with target PPM
        shift = peak_ppm - target_ppm

        # Apply the shift to the entire PPM scale
        adjusted_ppm = ppm - shift
        calibrated_X[i, :] = np.interp(ppm, adjusted_ppm, spectrum)

    calibrated_X_df = pd.DataFrame(calibrated_X)
    calibrated_X_df.columns = ppm
    calibrated_X_df.index = X.index

    return calibrated_X_df

class nmr_preprocessing:
    
    def __init__(self, data_path: str, bin_size: float = 0.0003, 
    auto_phasing: bool = True, baseline_correction: bool = True):
        self.data_path = data_path
        self.bin_size = bin_size
        self.auto_phasing = auto_phasing
        
        import os
        from glob import glob
        import tqdm
        import nmrglue as ng
        import pandas as pd
        
        # Check if the data path is valid
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path '{data_path}' not found.")
        

        #If last of data_path is / remove it
        if data_path.endswith('/'):
            data_path = data_path[:-1]
        
        #Get all subdirectories
        sub_dirs = glob(f'{data_path}/*/fid')
        #make dataframes
        import pandas as pd
        import numpy as np

        dir_ = pd.DataFrame(sub_dirs, columns = ['dir'])


        #Replace \ with /
        dir_['dir'] = dir_['dir'].apply(lambda x: x.replace('\\', '/'))


        #Split last part of the path to columns folder name

        dir_['folder name'] = dir_['dir'].apply(lambda x: x.split('/')[-2])


        dir_['dir'].replace('fid', '', regex=True, inplace=True)
        
        nmr_data = pd.DataFrame()
        
        for i in tqdm.tqdm(dir_.index):
            
            
            path_to_process = dir_['dir'][i]
            
            print(f'Processing file {path_to_process}')
            
            
            # Read in the Bruker formatted data
            try:
                dic, data = ng.bruker.read(path_to_process)
            except Exception as e:
                print(f"Error reading Bruker data: {e}")
                raise
            
            # Remove the digital filter
            data = ng.bruker.remove_digital_filter(dic, data)
                        # Generate PPM scale manually
            size = len(data)
            sweep_width = dic['acqus']['SW']            # Spectral width in ppm
            spectrometer_freq = dic['acqus']['SFO1']    # Spectrometer frequency in MHz
            offset = dic['procs']['OFFSET']                 # Offset in ppm


            # Process the spectrum
            #zf_size = 2 ** int(np.ceil(np.log2(len(data))))  # Next power of 2 for efficient FFT
            zf_size = int(sweep_width / bin_size)
            data = ng.proc_base.zf_size(data, zf_size)       # Zero fill
            data = ng.proc_base.fft(data)                    # Fourier transform
            data = ng.process.proc_autophase.autops(data, fn='peak_minima', p0=0.0, p1=0.0, return_phases=False)  # Phase correction
            #data = ng.process.proc_autophase.autops(data, fn='peak_minima')  # Phase correction


            # Discard the imaginaries and reverse data if needed
            data = ng.proc_base.di(data)
            data = ng.proc_base.rev(data)


            # Generate PPM scale
            ppm = np.linspace(offset, offset - sweep_width, zf_size)


            # Metadata
            print(f"Sweep Width: {sweep_width} ppm")
            print(f"Spectrometer Frequency: {spectrometer_freq} MHz")
            print(f"Offset: {offset} ppm")
            print(f"Data size: {data.size} data points")
            
            nmr_data = pd.concat([nmr_data, pd.DataFrame(data).T], axis=0)
            dic_array = np.array(list(dic.items()))
        
        try:
            nmr_data.index = dir_['folder name'].astype(int).to_list()
        except:
            nmr_data.index = dir_['folder name'].to_list()

        nmr_data.columns = ppm
        nmr_data.sort_index(inplace=True)

        


        text_completed = '''\n
                        -----------------------------------------\n
                        \n      
                                Data processing completed
                        \n
                        -----------------------------------------\n
                        '''
        print(text_completed)
        
        
        self.nmr_data = nmr_data
        self.ppm = ppm
        self.dic_array = dic_array
        
        
        
    def get_data(self):
        
        #flip the data
        nmr_data_2 = self.nmr_data.iloc[:, ::-1]
        return nmr_data_2


    def get_ppm(self):
        ppm = self.ppm
        return ppm

    def get_metadata(self):
        return self.dic_array


