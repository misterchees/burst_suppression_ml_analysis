import os
from typing import Tuple
import numpy as np
import pandas as pd
import scipy.io

from MachineLearning.IO.io_core import IOCore
from MachineLearning.Utils.path_utils import PathUtils


def load_psd_with_start_end_resultid(directory_path: str, filename: str) \
        -> Tuple[pd.DataFrame, int, int, int]:
    """
    Loads a PSD csv file as Dataframe and returns it with metadata from the filename
    :param filename: a file with this name structure -> start_end_resultid.csv
    :param directory_path: The directory of filename
    :return: A tuple structured this way (dataframe, start, end, result_id)
    """
    psd_fullpath = PathUtils.create_anypath(directory_path, filename)
    print(f"Processing {psd_fullpath}")
    # Validation of single episode PSD name structure
    name_parts = filename.split(".")[0].split("_")
    if len(name_parts) != 4 and name_parts[0] != "PSD":
        raise ValueError(f"Name of file {filename} has no typical structure for single episode PSD")
    metadata = filename.replace(".csv", "").split("_")[1:]  # PSD_0_1_2.csv -> ['0','1','2']
    start = int(metadata[0])
    end = int(metadata[1])
    result_id = int(metadata[2])
    psd_dataframe = pd.read_csv(psd_fullpath)

    return psd_dataframe, start, end, result_id


class LoadData(IOCore):
    def __init__(self):
        super().__init__()

    def load_faw_times_as_df(self, parameters: dict) -> pd.DataFrame:
        """
        Assembles a path to the csv-file of interest depending on passed parameters
        in the Fake-Awake (FAW) directory and loads it into a Pandas DataFrame.
        :param parameters: A dictionary with all episode parameters from the project
        :return: A pandas DataFrame containing the episodes based on the parameters passed.
        """
        faw_dir = self.level1_subdir_path("faw")
        csv_fullpath = PathUtils.return_csv_fullpath(faw_dir, "result", parameters)
        # validate fullpath
        if not os.path.isfile(csv_fullpath):
            raise FileNotFoundError(f"CSV not found: {csv_fullpath}")
        # read CSV to DataFrame
        df = pd.read_csv(csv_fullpath)
        return df

    def load_grouped_faw_times(self, parameters: dict) -> dict[int, list[tuple[int, int]]]:
        epoch_times_df = self.load_faw_times_as_df(parameters)
        grouped_epoch_times = self.group_epochs_by_result_id(epoch_times_df)
        return grouped_epoch_times

    def load_awake_times_as_df(self, parameters: dict) -> pd.DataFrame:
        """
        Reads a CSV file with 'caseid' and 'anestart' columns and generates epochs
        based on a fixed epoch length.

        :param parameters: Parameters for episodes. Contains length of each epoch.
        :returns: A DataFrame with columns ['Start', 'End', 'ResultID'] representing the epochs.
        """
        csv_path = self.return_file_from_basedir("awake_times")
        input_df = pd.read_csv(csv_path)
        epoch_length = int(parameters["fixed_window_size"])

        all_epochs = []

        for _, row in input_df.iterrows():
            caseid = row['caseid']
            anestart = row['anestart']
            num_epochs = int(anestart // epoch_length)  # segment into epochs based on episode length

            for i in range(num_epochs):
                start = i * epoch_length
                end = start + epoch_length
                all_epochs.append({
                    'Start': start,
                    'End': end,
                    'ResultID': caseid
                })

        return pd.DataFrame(all_epochs)

    def load_grouped_awake_times(self, parameters: dict) -> dict[int, list[tuple[int, int]]]:
        epoch_times_df = self.load_awake_times_as_df(parameters)
        grouped_epoch_times = self.group_epochs_by_result_id(epoch_times_df)
        return grouped_epoch_times

    def return_eeg_tuple(self, result_id: int, filtered=True) -> Tuple[int, np.ndarray]:
        """
        Assembles a path to the EEG File of interest, specified by the patient ID and
        returns fs and raw EEG as a Tuple

        :param result_id: The patient ID
        :param filtered: If True retrieves the filtered EEG file instead of the raw EEG file
        :return: a tuple (fs, eeg). fs -> sampling frequency; eeg -> an EEG samples array with two channels
        """
        if filtered:
            fs, eeg = self.return_filtered_eeg_tuple(result_id)
        else:
            fs, eeg = self.return_raw_eeg_tuple(result_id)
        return fs, eeg

    def return_raw_eeg_tuple(self, result_id: int) -> Tuple[int, np.ndarray]:
        """
        Assembles a path to the raw EEG .mat file of interest, specified by the patient ID and
        returns fs and raw EEG as a Tuple

        :param result_id: The patient ID
        :return: a tuple (fs, eeg). fs -> sampling frequency; eeg -> a raw-EEG samples array with two channels
        """

        # Assemble Path to directory with .mat files
        vitaldb_eeg_dir = self.level2_subdir_path("initial_data", "raw_eeg_mat")

        mat_file_path = os.path.join(vitaldb_eeg_dir, f"{result_id}.mat")
        if not os.path.isfile(mat_file_path):
            raise FileNotFoundError(f"MAT-file not found: {mat_file_path}")

        # load .mat file
        eeg_cols = self.data_names["eeg_files"]
        mat_data = scipy.io.loadmat(mat_file_path)
        fs = int(mat_data[eeg_cols["eeg_fs"]].squeeze())
        raw_eeg = mat_data[eeg_cols["eeg_rawEEG"]]

        return fs, raw_eeg

    def return_filtered_eeg_tuple(self, result_id: int) -> Tuple[int, np.ndarray]:
        """
        Loads a filtered EEG from a CSV file with a comment line containing the sampling rate (fs).
        :param result_id: The patient ID
        :returns: a tuple (fs, eeg). fs -> sampling frequency; eeg -> a filtered-EEG samples array with two channels
        """
        # Assemble Path to directory with .csv files
        filtered_eeg_dir = self.level1_subdir_path("filtered_data")

        filepath = os.path.join(filtered_eeg_dir, f"{result_id}.csv")
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            if not first_line.startswith("# fs = "):
                raise ValueError("First line does not contain sampling rate in expected format: '# fs = <int>'")
            fs = int(first_line.replace("# fs = ", "").strip())

        # Load EEG data, skipping the first line (comment line)
        df = pd.read_csv(filepath, skiprows=1)
        eeg_cols = self.data_names["eeg_files"]
        raw_eeg = df[eeg_cols["eeg_channels"]].to_numpy()

        return fs, raw_eeg

    def read_eeg_epochs_from_csv(self, result_id: int, epochs: list, channel: int) -> tuple[int, dict]:
        """
        Reads only selected EEG segments (epochs) for a given channel from a CSV file with
        a header comment and sampling rate.

        :param result_id: Patient ID to determine path to the filtered EEG CSV file.
        :param epochs: List of (start_time, end_time) tuples in seconds.
        :param channel: EEG channel to extract (1 or 2).
        :returns: Tuple of (sampling rate as int, dict of (start, end) -> EEG segment as ndarray)
        """
        # Step 0: Assemble Path to directory with .csv files
        filtered_eeg_dir = self.level1_subdir_path("filtered_data")

        filepath = os.path.join(filtered_eeg_dir, f"{result_id}.csv")
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Step 1: Read sampling rate (fs) from the first comment line
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            if not first_line.startswith("# fs = "):
                raise ValueError("Missing or malformed sampling rate comment line.")
            fs = int(first_line.replace("# fs = ", "").strip())

        # Step 2: Compute required data row indices (account for header lines)
        index_offset = 2  # One comment line and one header line
        required_rows = set()
        sample_ranges = []

        for start, end in epochs:
            start_row = int(start * fs) + index_offset
            end_row = int(end * fs) + index_offset
            sample_ranges.append((start_row, end_row))
            required_rows.update(range(start_row, end_row))

        # Step 3: Build row-skipping function for pandas
        def skiprows(i):
            return i != 1 and i not in required_rows  # 1 is the header line, keep it

        # Step 4: Read only selected rows and the requested channel
        col = str(channel)  # 1 -> '1'
        df = pd.read_csv(filepath, usecols=[col], skiprows=skiprows)

        # Step 5: Extract segments from the DataFrame
        values = df[col].to_numpy()
        segments = {}
        cursor = 0

        # create output dict as segments ((start, end): segment)
        for (start_sec, end_sec), (start_row, end_row) in zip(epochs, sample_ranges):
            num_samples = end_row - start_row
            segment = values[cursor:cursor + num_samples]
            segments[(start_sec, end_sec)] = segment
            cursor += num_samples

        return fs, segments

    @staticmethod
    def group_epochs_by_result_id(all_epochs_df: pd.DataFrame) -> dict[int, list[tuple[int, int]]]:
        """
        Groups epoch start/end times by ResultID into a dictionary.

        :param all_epochs_df: DataFrame with columns 'Start', 'End', 'ResultID'
        containing epochs from different patients
        :returns: Dictionary {ResultID: [(start1, end1), (start2, end2), ...]}
        """
        grouped = (
            all_epochs_df.groupby("ResultID")[["Start", "End"]]
            .apply(lambda x: [tuple(row) for row in x.to_numpy()])
            .to_dict()
        )
        return grouped

    def return_file_from_basedir(self, file_key: str) -> str:
        """Returns a path to a file in base directory"""
        base_dir = self.path_config["base_dir"]["path_name"]
        file_name = self.path_config["base_dir"]["files"][file_key]
        file_path = PathUtils.create_anypath(base_dir, file_name)
        return file_path
