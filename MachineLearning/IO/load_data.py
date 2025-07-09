"""Module for the LoadData class, and a psd load function"""
import os
from typing import Tuple
import numpy as np
import pandas as pd

from MachineLearning.IO.io_core import IOCore
from MachineLearning.Utils.path_utils import PathUtils


def load_psd_with_start_end_resultid(folder_path: str, filename: str) \
        -> Tuple[pd.DataFrame, int, int, int]:
    """
    Loads a PSD csv file from given folder as Dataframe and returns it with metadata from the filename
    :param filename: a file with this name structure -> start_end_resultid.csv
    :param folder_path: The path to the folder of filename
    :return: A tuple structured this way (dataframe, start, end, result_id)
    """
    psd_fullpath = PathUtils.return_anypath(folder_path, filename)
    print(f"Processing {psd_fullpath}")
    # Validation of single episode PSD name structure
    name_parts = filename.split(".")[0].split("_")
    if len(name_parts) != 4 and name_parts[0] != "PSD":
        raise ValueError(f"Name of file {filename} has no typical structure for single episode PSD."
                         "Typical structure example: PSD_0_1_2.csv")
    metadata = filename.replace(".csv", "").split("_")[1:]  # PSD_0_1_2.csv -> ['0','1','2']
    start = int(metadata[0])
    end = int(metadata[1])
    result_id = int(metadata[2])
    psd_dataframe = pd.read_csv(psd_fullpath)

    return psd_dataframe, start, end, result_id


class LoadData(IOCore):
    """This class handles the loading of the data in the project"""

    def __init__(self):
        """Initializes the class with configs from IOCore superclass."""
        super().__init__()

    def load_faw_times_as_df(self, parameters: dict) -> pd.DataFrame:
        """
        Assembles a path to the csv-file of interest depending on passed parameters
        in the Fake-Awake (FAW) directory and loads it into a Pandas DataFrame.
        :param parameters: A dictionary with all episode parameters from the project
        :return: A pandas DataFrame containing the episodes based on the parameters passed.
        """
        faw_dir = self.return_folder_path("faw")
        parameter_dir = PathUtils.return_A_B_C_D_X_Y_path("result", parameters)
        csv_fullpath = PathUtils.return_anypath(faw_dir, f"{parameter_dir}.csv")
        # validate fullpath
        if not os.path.isfile(csv_fullpath):
            raise FileNotFoundError(f"CSV not found: {csv_fullpath}")
        # read CSV to DataFrame
        df = pd.read_csv(csv_fullpath)
        return df

    def load_awake_times_as_df(self, parameters: dict, transition_time=10) -> pd.DataFrame:
        """
        Reads a CSV file with 'caseid' and 'anestart' columns and generates epochs
        based on a fixed epoch length.

        :param parameters: Parameters for episodes. Contains length of each epoch.
        :param transition_time: The transition time for patient to respond to anesthesia beginning.
        :returns: A DataFrame with columns ['Start', 'End', 'ResultID'] representing the epochs.
        """
        csv_path = self.return_csv_path_from_basedir("awake_times")
        input_df = pd.read_csv(csv_path)
        epoch_length = int(parameters["fixed_window_size"])

        all_epochs = []

        for _, row in input_df.iterrows():
            caseid = row['caseid']
            anestart = int(row['anestart'])
            num_epochs = int(
                (anestart - transition_time) // epoch_length)  # segment into epochs based on episode length

            for i in range(num_epochs):
                start = i * epoch_length
                end = start + epoch_length
                all_epochs.append({
                    'Start': start,
                    'End': end,
                    'ResultID': caseid
                })

        return pd.DataFrame(all_epochs)

    def sample_anesthesia_epochs(self, parameters: dict, num_epochs: int, transition_sec: int = 10,
                                 safety_margin_min: int = 10, random_state: int = 42,
                                 epochs_per_eeg: int = 30) -> pd.DataFrame:
        """
        Samples random EEG epochs from anesthesia segments (i.e., neither awake nor FAW).

        :param parameters: Defines length of each epoch.
        :param num_epochs: Number of total epochs to sample
        :param transition_sec: Time (in seconds) after anestart before epochs are allowed
        :param safety_margin_min: Minutes to exclude from end of EEG to avoid flatline segments
        :param random_state: Random seed for reproducibility
        :param epochs_per_eeg: Max number of epochs to sample per eeg (Less samples per patient -> better distribution)
        :returns: DataFrame with columns Start, End, ResultID
        """
        import random

        # set initial data
        random.seed(random_state)
        anestart_csv_path = self.return_csv_path_from_basedir("awake_times")
        epoch_length_sec = int(parameters["fixed_window_size"])
        filtered_data_dir = self.return_folder_path("filtered_data")

        anestart_df = pd.read_csv(anestart_csv_path)
        result = []

        # List of all available EEG files
        eeg_files = [f for f in os.listdir(filtered_data_dir) if f.endswith('.csv')]
        random.shuffle(eeg_files)

        for eeg_file in eeg_files:
            result_id = os.path.splitext(eeg_file)[0]  # 1.csv -> ['1', '.csv'] -> 1
            eeg_path = os.path.join(filtered_data_dir, eeg_file)
            # Lookup anestart for patient ID
            match = anestart_df.loc[anestart_df['caseid'].astype(str) == result_id, 'anestart']
            if not match.empty:
                anestart = match.values[0]
            else:
                # Default to safety margin if anestart is not available for this ID
                anestart = safety_margin_min * 60

            # Get duration of EEG file
            with open(eeg_path) as f:
                num_lines = sum(1 for _ in f) - 1  # minus header
            eeg_duration = num_lines // 128  # assuming 128 Hz sampling rate (128 rows per second)

            # Compute valid range for sampling
            start_limit = anestart + transition_sec
            end_limit = eeg_duration - (safety_margin_min * 60) - epoch_length_sec
            if end_limit <= start_limit:
                continue  # skip files where there's not enough space

            # Skip files that are too short to retrieve an epoch
            max_possible_epochs = (end_limit - start_limit) // epoch_length_sec
            if max_possible_epochs <= 0:
                continue

            # Sample max is constrained by length of EEG, number of needed epochs, or given number per EEG
            # The smallest of these three will be picked
            num_to_sample = min(max_possible_epochs, num_epochs - len(result), epochs_per_eeg)
            start_points = random.sample(range(start_limit, end_limit - epoch_length_sec + 1), num_to_sample)

            for start in start_points:
                result.append({
                    "Start": start,
                    "End": start + epoch_length_sec,
                    "ResultID": result_id
                })

            if len(result) >= num_epochs:
                break  # early exit if we've collected enough

        return pd.DataFrame(result)

    def return_grouped_epochs(self, parameters: dict, epoch_type: str, num_epochs: int = None
                              ) -> dict[int, list[tuple[int, int]]]:
        """
        Returns EEG snippets (Epochs) based on given parameters.
        :param parameters: Parameters for episodes containing primary parameters for chosen faw eopchs.
        :param epoch_type: Defines epoch times.
        :param num_epochs: Number of anesthesia epochs. Will be ignored if epoch_type is not 'normal_an'
        :return: Dictionary {ResultID: [(start1, end1), (start2, end2), ...]}
        """
        # retrieve epochs based on epoch type
        if epoch_type == 'awake':
            epoch_times_df = self.load_awake_times_as_df(parameters)
        elif epoch_type == 'faw':
            epoch_times_df = self.load_faw_times_as_df(parameters)
        elif epoch_type == 'normal_an':
            epoch_times_df = self.sample_anesthesia_epochs(parameters, num_epochs=num_epochs)
        else:
            raise ValueError(f'Epoch type "{epoch_type}" not recognized. Valid options: "awake", "faw", "normal_an"')

        # Group and return epochs
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
        # Lazy import
        import scipy.io

        # Assemble Path to directory with .mat files
        vitaldb_eeg_dir = self.return_folder_path("initial_data", "raw_eeg_mat")

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
        filtered_eeg_dir = self.return_folder_path("filtered_data")

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
        filtered_eeg_dir = self.return_folder_path("filtered_data")

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
            return i != 1 and i not in required_rows  # only keeps header line and required rows

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

    def return_csv_path_from_basedir(self, file_key: str) -> str:
        """Returns a path to a file in the base directory"""
        base_dir = self.path_config["base_dir"]["path_name"]
        file_name = self.path_config["base_dir"]["files"][file_key]
        file_path = PathUtils.return_anypath(base_dir, f"{file_name}.csv")
        return file_path

    def load_model(self, model_key: str, parameters: dict):
        """
        Loads a model from a specified path, defined by model_key and parameters.
        :param model_key: Key of the model (folder)
        :param parameters: Parameters for the model -> define the subfolder names.
        """

        from joblib import load
        full_folder_path = self.return_all_parameter_fullpath(parameters, False, True, "models", model_key)
        model_file = f"{model_key}.joblib"
        model_fullpath = PathUtils.return_anypath(full_folder_path, model_file)
        model = load(model_fullpath)
        return model

    def load_metrics(self, parameters: dict, model_key: str, ):
        # Construct the path to the folder from where the file will be loaded
        folder_path = self.return_all_parameter_fullpath(parameters, False, True, "results", model_key)
        file_name = f"folds_metrics.json"
        fullpath = PathUtils.return_anypath(folder_path, file_name)

        return PathUtils.load_json(fullpath)

