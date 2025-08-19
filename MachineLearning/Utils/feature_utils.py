import pandas as pd
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Preprocessing.normalizing import Normalizing


class FeatureUtils:
    @staticmethod
    def combine_features(parameters: dict, epoch_type: str, all_features: bool, features: list, normalize=True):
        """
        Combines multiple feature CSVs (based on ResultID, Start, End) into a single DataFrame
        and saves it to the feature_sets directory.

        :param parameters: Parameters of the episodes from which the features are. Defines subfolder of features.
        :param all_features: If True, combine all available features found in the features directory.
        :param epoch_type: Defines the epochs from which the features were calculated. Defines subfolder of features.
        :param features: Specific feature keys to include (used only if all_features=False).
        """
        loader = LoadData()
        saver = SaveResult()
        # Step 1: Determine which features to include
        if all_features:
            features = loader.return_all_feature_keys()
        else:
            if not features:
                raise ValueError("You must provide at least one feature name or set all_features=True.")

        # Remove PSDs if present since it is just the basis of the features, but no feature itself
        if "psds" in features:
            features.remove("psds")

        # Step 2: Load feature CSVs and merge on Start, End, ResultID
        merged_df = None

        for feature in features:
            try:
                print(f"Merging feature {loader.return_feature_name(feature)}...")
                # loads feature
                feature_path = loader.return_file_fullpath(parameters, True, False, epoch_type, ["features", feature])
                df = pd.read_csv(feature_path)
                # Merge or initialize
                if merged_df is None:
                    merged_df = df
                else:
                    # Only keep one copy of Start, End, ResultID (must match)
                    merged_df = pd.merge(merged_df, df, on=["Start", "End", "ResultID"], how="inner")
            except FileNotFoundError:
                print(FileNotFoundError)
            except Exception as e:
                print(f"An error occured while merging feature {feature}: {e}")

        # Check if the merged file is empty
        if merged_df is None or merged_df.empty:
            raise ValueError("No features were combined – possibly no files found or empty inputs.")

        # Drop NaN rows, if present
        merged_df = merged_df.dropna()

        # z-score normalization
        normalizer = Normalizing("zscore")
        norm_func = normalizer.normalize_array
        merged_df = FeatureUtils._normalize_features_in_df(merged_df, norm_func)

        # Step 3: Sort for consistency
        merged_df = merged_df.sort_values(by=["ResultID", "Start", "End"]).reset_index(drop=True)

        # Step 4: Save to the feature_sets directory
        saver.save_combined_features(parameters, merged_df, epoch_type)

    @staticmethod
    def return_eeg_epochs(epoch_type: str, parameters: dict, channel=1, num_an: int = None, allowed_ids: list = None) -> list:
        """
        Takes parameters and returns a list of all epochs (+ metadata) of this list

        :param epoch_type: Type of epoch. Influences from where to retrieve the epochs.
        :param parameters: Defines the directory from where the episodes will be retrieved.
        :param channel: EEG-Channel (options: 1, 2)
        :param num_an: Number of epochs for normal anesthesia to return. Only relevant if epoch_type = 'normal_an'
        :param allowed_ids: List of patient IDs to return. If None, all IDs are returned.
        :return: List of Tuples. Every Tuple is structured -> (start(s), end(s), result_id, fs, eeg epochs (samples))
        """

        data_loader = LoadData()
        output_list = []

        # Return Epoch times based on current parameters and grouped by result ID
        episode_times_df = data_loader.load_grouped_epochs(parameters, epoch_type, num_an)
        if epoch_type != 'normal_an':
            print(f"Retrieving Epochs for {epoch_type} for Parameters: {parameters}")
        else:
            print(f"Sampling {num_an} Epochs from normal anesthesia")

        if allowed_ids is not None:
            episode_times_df = {pid: df for pid, df in episode_times_df.items() if pid in allowed_ids}

        for result_id, epoch_list in episode_times_df.items():
            # get times, segments and fs from grouped times list
            fs, eeg_segment_dict = data_loader.load_eeg_epochs_from_csv(
                result_id, epoch_list, channel, ["normalized_data"]
            )

            # Assemble tuple start, end, result, fs, eeg epoch -> Add it to list
            for times, eeg_segment in eeg_segment_dict.items():
                start_time, end_time = times
                data_tuple = (start_time, end_time, result_id, fs, eeg_segment)
                output_list.append(data_tuple)
                print(f"Epoch for Patient ID {result_id}: Start time {start_time}, End time: {end_time}")

        return output_list

    @staticmethod
    def _normalize_features_in_df(df, normalizing_function):
        """
        Normalize all feature columns in a DataFrame column-wise using a given normalization function.
        Keeps Start, End, ResultID unchanged.

        :param df: pandas.DataFrame with columns ["Start", "End", "ResultID", feature1, feature2, ...]
        :param normalizing_function: function that takes a numpy array and returns a normalized numpy array
        :return: pandas.DataFrame with normalized features
        """
        meta_cols = ["Start", "End", "ResultID"]
        feature_cols = [col for col in df.columns if col not in meta_cols]

        df_norm = df.copy()
        for col in feature_cols:
            df_norm[col] = normalizing_function(df[col].values)

        return df_norm

    @staticmethod
    def return_all_features_dict() -> dict:
        """Returns all features as a dictionary"""
        data_loader = LoadData()
        return data_loader.path_config["base_dir"]["subdirs"]["features"]["subdirs"]
