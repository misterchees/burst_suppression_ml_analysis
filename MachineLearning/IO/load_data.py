import os
from typing import Tuple
import numpy as np
import pandas as pd
import scipy.io

import MachineLearning.IO.io_core as io_core
from MachineLearning.Utils.path_utils import PathUtils


def load_psd_with_start_end_resultid(directory_path: str, filename: str)\
        -> Tuple[pd.DataFrame, int, int, int]:
    """
    Loads a PSD csv file as Dataframe and returns it with metadata from the filename
    :param filename: a file with this name structure -> start_end_resultid.csv
    :param directory_path: The directory of filename
    :return: A tuple structured this way (dataframe, start, end, result_id)
    """
    psd_fullpath = PathUtils.create_anypath(directory_path, filename)
    print(f"Processing {psd_fullpath}")
    metadata = filename.replace(".csv", "").split("_")[1:]  # PSD_0_1_2.csv -> ['0','1','2']
    start = int(metadata[0])
    end = int(metadata[1])
    result_id = int(metadata[2])
    psd_dataframe = pd.read_csv(psd_fullpath)

    return psd_dataframe, start, end, result_id


class LoadData(io_core.IOCore):
    metadata_filename = "metadata_vitaldb.csv"
    combined_raw_data_subdir = "vitaldb_csvprocessed_BIS_BIS_SR_MAC"
    raw_eeg_mat_subdir = "vitalDB_mat_EEG"

    def __init__(self):
        super().__init__()

    def load_faw_csv_as_df(self, parameters: dict) -> pd.DataFrame:
        """
        Assembles a path to the csv-file of interest depending on passed parameters
        in the Fake-Awake (FAW) directory and loads it into a Pandas DataFrame.
        :param parameters: A dictionary with all episode parameters from the project
        :return: A pandas DataFrame containing the episodes based on the parameters passed.
        """
        faw_dir = self.create_faw_path()
        csv_fullpath = PathUtils.create_csv_fullpath(faw_dir, "result", parameters)
        # validate fullpath
        if not os.path.isfile(csv_fullpath):
            raise FileNotFoundError(f"CSV not found: {csv_fullpath}")
        # read CSV to DataFrame
        df = pd.read_csv(csv_fullpath)
        return df

    def return_eeg_tuple(self, result_id: int) -> Tuple[int, np.ndarray]:
        """
        Assembles a path to the EEG mat File of interest, specified by the patient ID and
        returns fs and raw EEG as a Tuple

        :param result_id: The patient ID
        :return: a tuple containing the sampling frequency and an array with two channels of raw EEG
        """

        # Assemble Path to directory with .mat files
        vitaldb_eeg_dir = self.create_mat_eeg_dir()

        mat_file_path = os.path.join(vitaldb_eeg_dir, f"{result_id}.mat")
        if not os.path.isfile(mat_file_path):
            raise FileNotFoundError(f"MAT-file not found: {mat_file_path}")

        # load .mat file
        mat_data = scipy.io.loadmat(mat_file_path)
        fs = int(mat_data[self.eeg_fs].squeeze())
        raw_eeg = mat_data[self.eeg_rawEEG]

        return fs, raw_eeg

    def create_mat_eeg_dir(self):
        return PathUtils.create_anypath(self.data_dir, self.initial_data_subdir, self.raw_eeg_mat_subdir)
