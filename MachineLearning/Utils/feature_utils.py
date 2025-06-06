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
        :return: List of Tuples. Every Tuple is structured -> (start(s), end(s), result_id, fs, eeg epoch (samples))
        """

        data_loader = self.data_loader
        output_list = []

        # Load FAW Episode based on current parameters
        episode_times_df = data_loader.load_faw_csv_as_df(parameters)

        for _, row in episode_times_df.iterrows():
            start_time = int(row['Start'])
            end_time = int(row['End'])
            result_id = int(row['ResultID'])

            # get eeg data for episodes in Patients record (defined by result_id)
            fs, raw_eeg = data_loader.return_eeg_tuple(result_id, filtered)

            # validate channel
            if channel not in [1, 2]:
                raise ValueError(f"Channel value is: {channel} but must be 1 or 2")

            eeg_signal = raw_eeg[:, channel - 1]

            # timeframe in samples
            start_sample = int(start_time * fs)
            end_sample = int(end_time * fs)
            eeg_segment = eeg_signal[start_sample:end_sample]

            # Assemble tuple start, end, result, fs, eeg epoch -> Add it to list
            data_tuple = (start_time, end_time, result_id, fs, eeg_segment)
            output_list.append(data_tuple)

        return output_list

    def return_all_features_dict(self):
        return self.data_loader.path_config["base_dir"]["subdirs"]["features"]["subdirs"]
