import numpy as np
import os
import pandas as pd
from MachineLearning.IO.io_core import IOCore
from MachineLearning.Core.utils import Utils


class SaveResult(IOCore):
    def __init__(self):
        super().__init__()

    def save_psd(self, frequencies: np.ndarray, power: np.ndarray,
                 parameters: dict, start: int, end: int, result_id: int):
        """
        Saves PSD data of EEG in a directory specified by parameters with a name specified by start, end and result_id.
        :param frequencies: Frequencies from the PSD
        :param power: Power of the Frequencies from the PSD
        :param parameters: A dictionary with all episode parameters from the project
        :param start: Start time of episode, from which the PSD was calculated
        :param end: End time of episode, from which the PSD was calculated.
        :param result_id: Patient ID
        """

        # assemble path to directory
        psd_dir = Utils.create_anypath(self.data_dir, self.features_subdir, self.psds_subdir)
        abcd_subdir = Utils.create_A_B_C_D_subfolder_name("PSD", parameters)
        xy_subdir = Utils.create_X_Y_subfolder_name(parameters)
        psd_dir_fullpath = Utils.create_anypath(psd_dir, abcd_subdir, xy_subdir)

        # create directory
        os.makedirs(psd_dir_fullpath, exist_ok=True)

        # create dataframe from PSD data
        psd_df = pd.DataFrame({
            self.psd_freq_col: frequencies,
            self.psd_power_col: power
        })

        # create fullpath with PSD name to save data
        psd_filename = f"PSD_{start}_{end}_{result_id}.csv"
        fullpath = Utils.create_anypath(psd_dir_fullpath, psd_filename)

        # save psd
        psd_df.to_csv(fullpath, index=False)

        print(f"Saved: {fullpath}")





