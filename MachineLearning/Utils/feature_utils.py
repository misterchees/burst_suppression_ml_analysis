from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult


class FeatureUtils:
    data_loader = LoadData()
    result_saver = SaveResult()

    def combine_features(self):
        pass

    def return_faw_eeg_epochs(self, parameters: dict, filtered=True, channel=1) -> list:
        """
        Takes parameters for fake awakeness (faw) and returns a list of all epochs (+ metadata) of this list

        :param parameters: Defines the directory from where the episodes will be retrieved.
        :param filtered: Defines if windows are from filtered EEG (True) or raw EEG (False).
        :param channel: EEG-Channel (options: 1, 2)
        :return: List of Tuples. Every Tuple is structured -> (start(s), end(s), result_id, fs, eeg epochs (samples))
        """

        data_loader = self.data_loader
        output_list = []

        # Load FAW Episode times based on current parameters and grouped by result ID
        episode_times_df = data_loader.load_grouped_faw_times(parameters)
        print(f"Retrieving Epochs for Parameters: {parameters}")

        for result_id, epoch_list in episode_times_df.items():
            # get times, segments and fs from grouped times list
            fs, eeg_segment_dict = data_loader.read_eeg_epochs_from_csv(result_id, epoch_list, channel)

            # Assemble tuple start, end, result, fs, eeg epoch -> Add it to list
            for times, eeg_segment in eeg_segment_dict.items():
                start_time, end_time = times
                data_tuple = (start_time, end_time, result_id, fs, eeg_segment)
                output_list.append(data_tuple)
                print(f"Epoch for Patient ID {result_id}: Start time {start_time}, End time: {end_time}")

        return output_list

    def return_all_features_dict(self):
        return self.data_loader.path_config["base_dir"]["subdirs"]["features"]["subdirs"]
