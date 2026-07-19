# -*- coding: utf-8 -*-

__author__ = 'aeiwz'
__email__ = 'theerayut_aeiw_123@hotmail.com'
__copyright__="Copyright 2024, Theerayut"
__license__ = "MIT"
__maintainer__ = "aeiwz"
__status__ = "Development"

import logging

logger = logging.getLogger(__name__)

def read_fid(data_path: str):  # pragma: no cover
    """Read a Bruker FID file from the given directory.

    Args:
        data_path: Path to the Bruker FID directory.

    Returns:
        tuple: (dic, data) where dic is the parameter dictionary and
            data is the raw FID array.

    Examples:
        >>> dic, data = read_fid('data/sample_001')
        >>> print(type(data))
        <class 'numpy.ndarray'>
    """
    import nmrglue as ng
    logger.debug("Reading FID from: %s", data_path)
    dic, data = ng.bruker.read(data_path)
    return dic, data

def remove_digital_filter(dic, data):  # pragma: no cover
    """Remove the Bruker digital filter from the FID data.

    Args:
        dic: Bruker parameter dictionary returned by read_fid.
        data: Raw FID array returned by read_fid.

    Returns:
        numpy.ndarray: FID data with the digital filter removed.

    Examples:
        >>> dic, data = read_fid('data/sample_001')
        >>> data = remove_digital_filter(dic, data)
        >>> print(data.shape)
        (65536,)
    """
    import nmrglue as ng
    logger.debug("Removing digital filter")
    data = ng.bruker.remove_digital_filter(dic, data)
    return data

def generate_ppm_scale(dic, data):  # pragma: no cover
    """Generate a PPM scale array for a processed NMR spectrum.

    Args:
        dic: Bruker parameter dictionary containing acquisition parameters.
        data: Processed spectrum array (used only to determine the number
            of points).

    Returns:
        numpy.ndarray: PPM values corresponding to each data point,
            running from high to low field.

    Examples:
        >>> dic, raw = read_fid('data/sample_001')
        >>> raw = remove_digital_filter(dic, raw)
        >>> ppm = generate_ppm_scale(dic, raw)
        >>> print(ppm[0], ppm[-1])
        11.9873 -3.0124
    """
    import numpy as np
    size = len(data)
    sweep_width = dic['acqus']['SW']
    spectrometer_freq = dic['acqus']['SFO1']
    offset = dic['procs']['OFFSET']
    logger.debug("SW=%s, SFO1=%s, OFFSET=%s, size=%s", sweep_width, spectrometer_freq, offset, size)
    ppm = np.linspace(offset, offset - sweep_width, size)
    return ppm

def phasing(data, index, auto=True, fn='peak_minima', p0=0.0, p1=0.0):  # pragma: no cover
    """Apply phase correction to an NMR spectrum at the given index.

    Args:
        data: 2-D array of spectra where rows are individual spectra.
        index: Row index of the spectrum to phase.
        auto: If True, automatic phasing is applied using nmrglue
            autops (default: True).
        fn: Algorithm name passed to autops (default: 'peak_minima').
        p0: Zero-order phase correction in degrees (default: 0.0).
        p1: First-order phase correction in degrees (default: 0.0).

    Returns:
        numpy.ndarray: The data array with the spectrum at *index*
            phase-corrected in place.

    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(5, 65536) + 1j * np.random.randn(5, 65536)
        >>> data = phasing(data, index=0, auto=True, fn='peak_minima')
        >>> print(data.shape)
        (5, 65536)
    """
    import nmrglue as ng
    logger.debug("Phasing spectrum at index %s, auto=%s, fn=%s", index, auto, fn)
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
    def __init__(self, data_path: str, bin_size: float = 0.0003,  # pragma: no cover
                auto_phasing: bool = False, fn_ = 'acme',
                baseline_correction: bool = True, baseline_type: str = 'linear',
                calibration: bool = True, calib_type: str = 'tsp',
                custom_range: tuple | None = None, custom_target: float | None = None,
                align: bool = False, align_reference: str = 'median',
                align_max_shift_ppm: float = 0.02, align_top_n: int = 30,
                align_windows: list[tuple[float, float]] | None = None):

        self.data_path = data_path
        self.bin_size = bin_size
        self.auto_phasing = auto_phasing
        self.baseline_correction = baseline_correction
        self.calibration = calibration
        self.calib_type = calib_type
        self.baseline_type = baseline_type
        self.fn_ = fn_
        self.custom_range = custom_range
        self.custom_target = custom_target
        self.align = align
        self.align_reference = align_reference
        self.align_max_shift_ppm = align_max_shift_ppm
        self.align_top_n = align_top_n
        self.align_windows = align_windows

        import os
        from glob import glob
        import tqdm
        import nmrglue as ng
        import pandas as pd
        import numpy as np

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

        if not sub_dirs:
            raise ValueError(f"No 'fid' files found in '{data_path}'. Please check your folder structure.")

        logger.debug("Found %d fid directories", len(sub_dirs))

        dir_ = pd.DataFrame(sub_dirs, columns=['dir'])
        dir_['dir'] = dir_['dir'].apply(lambda x: x.replace('\\', '/'))
        dir_['folder name'] = dir_['dir'].apply(lambda x: x.split('/')[-2])
        # Strip only the trailing 'fid' filename (each match ends in '/fid'),
        # not every 'fid' substring, so paths like '.../myfid_data/...' survive.
        dir_['dir'] = dir_['dir'].str.replace(r'fid$', '', regex=True)

        nmr_data = pd.DataFrame()
        phase_data = pd.DataFrame(columns=['p0', 'p1'])
        dic_list = {}
        ppm = None

        dir_.sort_values('folder name', inplace=True)
        for i in tqdm.tqdm(dir_.index):
            path_to_process = dir_['dir'][i]
            logger.debug("Processing sample %s: %s", i, path_to_process)

            try:
                dic, data = ng.bruker.read(path_to_process)
            except Exception as e:
                logger.error("Reading Bruker data failed: %s", e)
                raise

            data = ng.bruker.remove_digital_filter(dic, data)
            sweep_width = dic['acqus']['SW']
            spectrometer_freq = dic['acqus']['SFO1']
            offset = dic['procs']['OFFSET']

            zf_size = int(sweep_width / bin_size)
            data = ng.proc_base.zf_size(data, zf_size)
            data = ng.proc_base.fft(data)

            if auto_phasing:
                data, (p0, p1) = ng.process.proc_autophase.autops(data, fn=fn_, return_phases=True)
                phase = [p0, p1]
                logger.debug("Auto phase angles: p0=%s, p1=%s", p0, p1)
            else:
                logger.debug("Manual phasing with p0=%s and p1=%s", dic['procs']['PHC0'], dic['procs']['PHC1'])
                data = ng.process.proc_base.ps(data, p0=dic['procs']['PHC0'], p1=dic['procs']['PHC1'], inv=True)
                phase = [dic['procs']['PHC0'], dic['procs']['PHC1']]

            data = ng.proc_base.di(data)

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

            ppm = np.linspace(offset, offset - sweep_width, zf_size)

            nmr_data = pd.concat([nmr_data, pd.DataFrame([data.real])], axis=0)
            dic_list.update({dir_['folder name'][i]: dic})
            phase_data = pd.concat([phase_data, pd.DataFrame(phase).T], axis=0)

        ppm_columns = ppm[::-1]
        nmr_data.columns = ppm_columns
        try:
            nmr_data.index = dir_['folder name'].astype(int).to_list()
        except Exception:
            nmr_data.index = dir_['folder name'].to_list()
        phase_data.index = nmr_data.index

        from .calibrate import calibrate
        if self.calibration:
            ppm_for_cal = nmr_data.columns.to_numpy(dtype=float)
            self.nmr_data2 = calibrate(
                nmr_data,
                ppm_for_cal,
                calib_type,
                custom_range=self.custom_range,
                custom_target=self.custom_target,
            )

        logger.info("Data processing completed")
        if self.calibration:
            self.nmr_data = self.nmr_data2
        else:
            self.nmr_data = nmr_data

        self.ppm = self.nmr_data.columns.to_numpy(dtype=float)
        self.dic_array = dic_list
        self.phase_data = phase_data

        if self.align:
            try:
                from .alignment import PeakAligner
                pa = PeakAligner(self.nmr_data, self.ppm, sf_mhz=float(spectrometer_freq))
                if self.align_windows is None:
                    windows, mptable = pa.auto_windows(top_n=self.align_top_n)
                    self.alignment_windows_ = windows
                    self.alignment_multiplets_ = mptable
                else:
                    windows = self.align_windows
                    self.alignment_windows_ = windows
                X_aligned, shift_map = pa.align(windows, reference=self.align_reference, max_shift_ppm=self.align_max_shift_ppm)
                self.nmr_data = X_aligned
                self.alignment_shifts_ = shift_map
                self.ppm = self.nmr_data.columns.to_numpy(dtype=float)
                logger.debug("Alignment completed with %d windows", len(windows))
            except Exception as e:
                logger.error("Alignment failed: %s", e)


    def get_data(self, flip_data=True):
        """Return the processed NMR data as a DataFrame.

        Args:
            flip_data: If True (default), reverse the column order so that
                the PPM axis runs from low to high field (left to right).

        Returns:
            pd.DataFrame: Processed NMR spectra with PPM values as column
                headers and sample identifiers as the index.

        Examples:
            >>> nmr = nmr_preprocessing('data/cohort_fids', calib_type='tsp')
            >>> spectra = nmr.get_data()
            >>> print(spectra.shape)
            (20, 39936)
        """
        nmr_data = self.nmr_data
        nmr_data.sort_index(inplace=True)
        if flip_data:
            nmr_data = nmr_data.iloc[:, ::-1]
        return nmr_data

    def get_ppm(self, flip_data=True):
        """Return the PPM scale array aligned with get_data().

        The returned axis is ordered to match the columns of
        get_data(flip_data=...) with the same flag, so plotting
        get_data() against get_ppm() pairs each intensity with its
        correct ppm (no mirror flip).

        Args:
            flip_data: Must match the flag passed to get_data(). With the
                default True the axis runs high-to-low field (high ppm on
                the left), the standard NMR display order.

        Returns:
            numpy.ndarray: 1-D array of PPM values for the columns of
                get_data(flip_data=...).

        Examples:
            >>> nmr = nmr_preprocessing('data/cohort_fids', calib_type='tsp')
            >>> ppm = nmr.get_ppm()
            >>> print(ppm.min(), ppm.max())
            -3.012 11.987
        """
        ppm = self.nmr_data.columns.to_numpy(dtype=float)
        return ppm[::-1] if flip_data else ppm

    def get_metadata(self):
        """Return the Bruker acquisition metadata for all processed samples.

        Returns:
            dict: Mapping of sample folder name to its nmrglue parameter
                dictionary (dic) from the Bruker read step.

        Examples:
            >>> nmr = nmr_preprocessing('data/cohort_fids', calib_type='tsp')
            >>> meta = nmr.get_metadata()
            >>> sample_key = list(meta.keys())[0]
            >>> print(meta[sample_key]['acqus']['SFO1'])
            600.13
        """
        return self.dic_array

    def get_phase(self):
        """Return the phase correction values applied to each spectrum.

        Returns:
            pd.DataFrame: DataFrame with columns 'p0' and 'p1' containing
                the zero-order and first-order phase angles (degrees) used
                for each sample.

        Examples:
            >>> nmr = nmr_preprocessing('data/cohort_fids', auto_phasing=True)
            >>> phase_df = nmr.get_phase()
            >>> print(phase_df.head())
               p0      p1
            1  12.3  -245.1
            2   9.8  -231.4
        """
        return self.phase_data

if __name__ == '__main__':  # pragma: no cover
    fid = 'dev/launch/data/test_nmr_data'
    nmr = nmr_preprocessing(fid, bin_size=0.0005, auto_phasing=False, fn_='acme',
                            baseline_correction=True, baseline_type='corrector',
                            calibration=True, calib_type='glucose')

    nmr_data = nmr.get_data()
    print(nmr_data.columns)
    from lingress import plot_NMR_spec
    fig = plot_NMR_spec(spectra=nmr_data, ppm=nmr.get_ppm(), label=None).single_spectra()
    fig.show()
