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
        faw_dir = self.return_folder_path(["faw"])
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
        filtered_data_dir = self.return_folder_path(["filtered_data"])

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

    def load_grouped_epochs(self, parameters: dict, epoch_type: str, num_epochs: int = None
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

    def load_eeg_data(self, result_id: int, filtered=True) -> Tuple[int, np.ndarray]:
        """
        Assembles a path to the EEG File of interest, specified by the patient ID and
        returns fs and raw EEG as a Tuple

        :param result_id: The patient ID
        :param filtered: If True retrieves the filtered EEG file instead of the raw EEG file
        :return: a tuple (fs, eeg). fs -> sampling frequency; eeg -> an EEG samples array with two channels
        """
        if filtered:
            fs, eeg = self._load_filtered_eeg_data(result_id)
        else:
            fs, eeg = self._load_raw_eeg_data(result_id)
        return fs, eeg

    def _load_raw_eeg_data(self, result_id: int) -> Tuple[int, np.ndarray]:
        """
        Assembles a path to the raw EEG .mat file of interest, specified by the patient ID and
        returns fs and raw EEG as a Tuple

        :param result_id: The patient ID
        :return: a tuple (fs, eeg). fs -> sampling frequency; eeg -> a raw-EEG samples array with two channels
        """
        # Lazy import
        import scipy.io

        # Assemble Path to directory with .mat files
        vitaldb_eeg_dir = self.return_folder_path(["initial_data", "raw_eeg_mat"])

        mat_file_path = os.path.join(vitaldb_eeg_dir, f"{result_id}.mat")
        if not os.path.isfile(mat_file_path):
            raise FileNotFoundError(f"MAT-file not found: {mat_file_path}")

        # load .mat file
        eeg_cols = self.data_names["eeg_files"]
        mat_data = scipy.io.loadmat(mat_file_path)
        fs = int(mat_data[eeg_cols["eeg_fs"]].squeeze())
        raw_eeg = mat_data[eeg_cols["eeg_rawEEG"]]

        return fs, raw_eeg

    def _load_filtered_eeg_data(self, result_id: int) -> Tuple[int, np.ndarray]:
        """
        Loads a filtered EEG from a CSV file with a comment line containing the sampling rate (fs).
        :param result_id: The patient ID
        :returns: a tuple (fs, eeg). fs -> sampling frequency; eeg -> a filtered-EEG samples array with two channels
        """
        # Assemble Path to directory with .csv files
        filtered_eeg_dir = self.return_folder_path(["filtered_data"])

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

    def load_eeg_epochs_from_csv(self, result_id: int, epochs: list, channel: int, folder_keys: list[str]) -> tuple[int, dict]:
        """
        Reads only selected EEG segments (epochs) for a given channel from a CSV file with
        a header comment and sampling rate.

        :param result_id: Patient ID to determine path to the EEG CSV file.
        :param epochs: List of (start_time, end_time) tuples in seconds.
        :param channel: EEG channel to extract (1 or 2).
        :param folder_keys: List of folder keys to determine the path to the EEG CSV file.
        :returns: Tuple of (sampling rate as int, dict of (start, end) -> EEG segment as ndarray)
        """
        # Step 0: Assemble Path to directory with .csv files
        eeg_dir = self.return_folder_path(folder_keys)

        filepath = os.path.join(eeg_dir, f"{result_id}.csv")
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
        """Takes the path_config key of a file in the base directory and returns a path to that file."""
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
        full_folder_path = self.return_all_parameter_fullpath(parameters, False, True, ["models", model_key])
        model_file = f"{model_key}.joblib"
        model_fullpath = PathUtils.return_anypath(full_folder_path, model_file)
        model = load(model_fullpath)
        return model

    def load_metrics(self, parameters: dict, model_key: str, run_name: str = None ):
        # Construct the path to the folder from where the file will be loaded
        folder_path = self.return_all_parameter_fullpath(
            parameters, False, True, ["results", model_key], run_name)
        file_name = f"folds_metrics.json"
        fullpath = PathUtils.return_anypath(folder_path, file_name)

        return PathUtils.load_json(fullpath)

    def load_metadata_file(self, parameters: dict, model_key: str, filename: str, outlier_run_name: str | None):
        """
        Loads metadata file based on the given parameters, model key, and filename.

        This method retrieves the file path for a metadata file using the specified
        parameters and determines its extension. The data is then loaded and returned
        if the file has a supported format.

        :param parameters:
            A dictionary containing the hyperparameters used to
            construct the file path.
        :param model_key:
            The key associated with a specific model, used as an identifier while
            forming the directory path for the metadata file.
        :param filename:
            The name of the metadata file to be loaded, including its extension.
        :param outlier_run_name:
            The name of the outlier run to load metadata for. If None, loads metadata for the current run_name.
        :return:
            Parsed data from the metadata file. The return type depends on the file
            format: a JSON object for `.json` files or a pandas DataFrame for `.csv`
            files.
        :rtype:
            Union[dict, pandas.DataFrame]
        :raises ValueError:
            If the file's extension is not supported (i.e., not "json" or "csv").
        """
        # Construct the path to the folder from where the file will be loaded
        folder_path = self.return_all_parameter_fullpath(
            parameters, False, False, ["metadata_analysis", model_key], outlier_run_name
        )
        fullpath = PathUtils.return_anypath(folder_path, filename)

        # Load file based on extension
        extension = filename.split(".")[-1]
        if extension == "json":
            return PathUtils.load_json(fullpath)
        elif extension == "csv":
            return pd.read_csv(fullpath)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")


    def load_problematic_ids(self, parameters: dict, model_key: str, outlier_run_name: str | None,
                             global_outliers: bool) -> list | None:
        """
        Loads the IDs of outlier groups related to a specific model, from a metadata file.
        If the specified file does not exist, attempts to create it from previous analysis results.

        :param parameters: A dictionary containing parameter configurations for the model.
        :type parameters: dict
        :param model_key: The key or identifier of the model for which problematic IDs are being loaded.
        :type model_key: str
        :param outlier_run_name: The name of the outlier run to load IDs for. If None, loads IDs for the current run_name.
        :type outlier_run_name: str | None
        :param global_outliers: Flag indicating whether to include outliers from all runs.
        :type global_outliers: bool
        :return: A list of outlier groups extracted from the metadata file.
        :rtype: list

        :raises FileNotFoundError: If the metadata file does not exist and cannot be created from
            previous analysis results.
        """

        print("Loading IDs of outlier groups...")
        try:
            outliers_df = self.load_metadata_file(
                parameters, model_key, "Summary_outliers_by_groups.csv", outlier_run_name
            )
        except FileNotFoundError:
            print("File 'Summary_outliers_by_groups.csv' not found. Trying to create from previous results...")
            try:
                from MachineLearning.Evaluation.meta_fold_analyzer import MetaFoldAnalyzer
                fold_analyzer = MetaFoldAnalyzer(model_key, parameters, outlier_run_name)
                outliers_df = fold_analyzer.select_outlier_groups(save_res=True)
            except FileNotFoundError:
                print("No previous results found to create 'Summary_outliers_by_groups.csv'. "
                      "No problematic IDs can be loaded.")
                return None

        # Ensure that global outliers include outliers from given run by saving them before loading global outliers
        if global_outliers:
            outliers_df = self._update_and_get_global_outliers(parameters, outliers_df, "patient_id")

        outlier_list = outliers_df["group"].values.tolist()
        print(f"Outlier groups that will be removed from analysis: {outlier_list}")

        return outlier_list

    def load_problematic_epochs(self, parameters: dict, model_key: str, outlier_run_name: str | None,
                                global_outliers: bool) -> pd.DataFrame | None:
        """
        Loads problematic epochs based on metadata or by analyzing previous results. If global outliers are
        requested, they will include the identified outliers from the current run before global outliers are loaded.
        The function returns the outlier epochs in the form of a DataFrame or None if no problematic epochs are found.

        :param parameters: Configuration parameters used in the operation.
        :type parameters: dict
        :param model_key: Unique identifier for the model being used.
        :type model_key: str
        :param outlier_run_name: Optional identifier for the run to load outlier metadata for.
        :type outlier_run_name: str | None
        :param global_outliers: Flag indicating whether to include global outliers in the analysis.
        :type global_outliers: bool
        :return: A DataFrame containing the problematic outlier epochs or None if no problematic epochs are found.
        :rtype: pd.DataFrame | None
        """

        print("Loading outlier epochs...")
        try:
            outliers_df = self.load_metadata_file(
                parameters, model_key, "Summary_outlier_epochs.csv", outlier_run_name
            )
        except FileNotFoundError:
            print("File 'Summary_outlier_epochs.csv' not found. Trying to create from previous results...")
            try:
                from MachineLearning.Evaluation.meta_fold_analyzer import MetaFoldAnalyzer
                fold_analyzer = MetaFoldAnalyzer(model_key, parameters, outlier_run_name)
                outliers_df = fold_analyzer.select_outlier_epochs(save_res=True)
            except FileNotFoundError:
                print("No previous results found to create 'Summary_outlier_epochs.csv'. "
                      "No problematic epochs can be loaded.")
                return None

        # Ensure that global outliers include outliers from given run by saving them before loading global outliers
        if global_outliers:
            outliers_df = self._update_and_get_global_outliers(parameters, outliers_df, "epoch")

        outlier_list = outliers_df.values.tolist()
        print(f"Outlier epochs that will be removed from analysis: {outlier_list}")

        return outliers_df

    def load_run_data(self, hyperparameters: dict, run_name: str, model_key: str):
        """Loads the run data from a run specified by name, hyperparameters, and model_key."""
        metadata_path = self.return_run_metadata_fullpath(hyperparameters, run_name, model_key)
        metadata = PathUtils.load_json(metadata_path)
        return metadata

    def load_splits(self, hyperparamers: dict, run_name: str, combined=True) -> dict|pd.DataFrame:
        """
        Loads dataset splits from specified file paths into a dictionary of DataFrames or a combined DataFrame.

        :param hyperparamers: Configuration dictionary specifying parameters
                              required to determine dataset split file paths.
        :type hyperparamers: dict
        :param run_name: Name of the run used to access associated split file paths.
        :type run_name: str
        :param combined: Flag to determine whether all splits should be combined
                         into a single DataFrame. Defaults to True.
        :type combined: bool
        :returns: A single combined DataFrame if 'combined' is True. Otherwise, a dictionary
                  of DataFrames indexed by the file name (stem) of each split.
        :rtype: dict | pd.DataFrame
        """
        splits_list = self.return_related_fullpaths(hyperparamers, run_name, ["test_and_train_data", "splits"])

        df_dict = {}
        for split_path in splits_list:
            df_dict[split_path.stem] = pd.read_csv(split_path)

        if combined:
            df_combined = pd.concat(df_dict.values(), axis=0)
            return df_combined
        else:
            return df_dict

    def load_results(self, hyperparamers: dict, run_name: str, model_key: str, combined=True) -> dict|pd.DataFrame:
        results_list = self.return_related_fullpaths(hyperparamers, run_name, ["results", model_key])

        df_dict = {}
        for result_path in results_list:
            df_dict[result_path.stem] = pd.read_csv(result_path)

        if combined:
            df_combined = pd.concat(df_dict.values(), axis=0)
            return df_combined
        else:
            return df_dict

    def load_combined_features_df(self, parameters: dict, class_1: str, class_0: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load combined features dataframes for the specified classes.

        This method loads data from specified file paths for two given classes and
        returns their corresponding dataframes. It uses the provided parameters to
        determine file paths for the data of `class_1` and `class_0`. The paths are
        generated dynamically based on the parameters and predefined folder hierarchy.

        :param parameters: A dict containing configuration parameters.
        :param class_1: The name or identifier of the first class whose data will be loaded.
            Valid options are: 'normal_an', 'faw' and 'awake'
        :param class_0: The name or identifier of the second class whose data will be loaded.
            Valic options are: 'normal_an', 'faw' and 'awake'
        :return: A tuple of pandas DataFrames corresponding to the data of `class_1` and `class_0`.

        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        class_1_path = self.return_file_fullpath(parameters, True, False, class_1,
                                                              ["test_and_train_data", "feature_sets"])
        class_0_path = self.return_file_fullpath(parameters, True, False, class_0,
                                                              ["test_and_train_data", "feature_sets"])
        print(f"Loading {class_1} data from {class_1_path}\n "
              f"Loading {class_0} data from {class_0_path}")
        class_1_df = pd.read_csv(class_1_path).copy()
        class_0_df = pd.read_csv(class_0_path).copy()

        return class_1_df, class_0_df

    def load_global_outliers(self, parameters, outlier_type) -> pd.DataFrame:
        """
        Loads a dataframe of global outliers based on the specified outlier type and parameters.
        The method determines the target file name based on the `outlier_type` and retrieves
        the file location using the provided parameters. It reads and returns the data from
        the corresponding CSV file.

        :param parameters:
            List of parameters used to determine the folder path for outliers files.
        :param outlier_type:
            Specifies the type of outliers to fetch, either "epoch" or "patient_id".
        :return:
            Pandas DataFrame containing the global outliers data.
        :rtype: pd.DataFrame
        :raises ValueError:
            If the provided outlier_type is not "epoch" or "patient_id".
        """
        folder_path = self.return_all_parameter_fullpath(parameters, False, False, ["global_outliers"])

        # Build fullpath depending on outlier type
        if outlier_type == "epoch":
            filename = "global_epoch_outliers.csv"
        elif outlier_type == "patient_id":
            filename = "global_patient_outliers.csv"
        else:
            raise ValueError("Invalid outlier type. Expected 'epoch' or 'patient_id'.")

        fullpath = PathUtils.return_anypath(folder_path, filename)
        # Load and return outlier df
        outliers_df = pd.read_csv(fullpath)
        return outliers_df

    def load_further_results(self, hyperparameters: dict, analysis_key: str, result_type: str, filename: str):
        folder_path = self.return_all_parameter_fullpath(hyperparameters, False, True,
                                                         ["further_analysis", analysis_key])

        if result_type == "dataframe":
            PathUtils.return_anypath(folder_path, filename)
            return pd.read_csv(PathUtils.return_anypath(folder_path, filename))
        elif result_type == "json":
            return PathUtils.load_json(PathUtils.return_anypath(folder_path, filename))
        else:
            raise ValueError(f"Invalid result type. Expected 'dataframe' or 'json', got {result_type}")

    def _update_and_get_global_outliers(self, parameters, outliers_df: pd.DataFrame, outlier_type: str):
        """
        Updates the global outliers with new outliers and retrieves the updated global outliers.
        This function saves the new outliers to the global outliers using an external saver instance
        and then fetches the updated global outliers dataset.

        :param parameters: The parameters required for saving and loading global outliers.
        :type parameters: Any
        :param outliers_df: A pandas DataFrame containing the new outliers to be added to the global outliers.
        :param outlier_type: A string representing the specific type of outlier
            valid options are: "patient_id", "epoch".
        :return: A pandas DataFrame containing the updated global outliers after addition of the new outliers.
        :rtype: pd.DataFrame
        """
        from MachineLearning.IO.save_result import SaveResult
        saver = SaveResult()
        saver.save_global_outliers(parameters, outliers_df, outlier_type)  # Add given outliers to global
        global_outliers_df = self.load_global_outliers(parameters, outlier_type)  # Get updated global outliers
        return global_outliers_df