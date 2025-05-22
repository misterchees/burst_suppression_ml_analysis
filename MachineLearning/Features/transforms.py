from MachineLearning.Core.ml_object import MLObject
from MachineLearning.IO.load_data import LoadData
from scipy.signal import welch


class Transform(MLObject):

    def __init__(self):
        super().__init__()

    def transform_eeg_to_psd(self, channel=1, nperseg_seconds=2):
        """
        Calculates PSDs for EEG windows in specified csv from preprocessing_csv_fullpath and saves
        every PSD in a seperate csv file in a defined output directory.

        Parameters:
        - channel: EEG-Channel (options: 1, 2)
        - nperseg_seconds: Length of window for Welch in seconds (usually: 1 or 2)
        """

        # Load FAW Episode based on current parameters
        input_dataframe = LoadData.load_faw_csv_as_df(self.parameter_dict)

        for _, row in input_dataframe.iterrows():
            result_id = int(row['ResultID'])
            start_time = int(row['Start'])
            end_time = int(row['End'])

            # unpack the data of .mat file of interest
            fs, raw_eeg = LoadData.return_eeg_tuple(result_id)

            # validate channel
            if channel not in [1, 2]:
                raise ValueError(f"Channel value is: {channel} but must be 1 or 2")

            eeg_signal = raw_eeg[:, channel - 1]

            # timeframe in samples
            start_sample = int(start_time * fs)
            end_sample = int(end_time * fs)
            eeg_segment = eeg_signal[start_sample:end_sample]

            # calculate welch PSD
            nperseg = int(nperseg_seconds * fs)
            frequencies, psd = welch(eeg_segment, fs=fs, nperseg=nperseg)

            # result as DataFrame
            psd_df = pd.DataFrame({
                self.psd_freq_col: frequencies,
                self.psd_power_col: psd
            })

            # create output directory with same structure as input subfolder: PSD_A_B_C_D\Summary_Episodes_X_Y
            psd_subfolder_1 = self.create_A_B_C_D_subfolder_name("PSD")
            psd_subfolder_2 = self.create_X_Y_subfolder_name()
            psd_output_path = os.path.join(self.output_dir, "PSDs", psd_subfolder_1, psd_subfolder_2)
            os.makedirs(psd_output_path, exist_ok=True)

            # save as PSD_H_K_L.csv according to structure in preprocessing CSV file. i.e. start, end, resultID
            psd_filename = f"PSD_{start_time}_{end_time}_{result_id}.csv"
            psd_file_path = os.path.join(psd_output_path, psd_filename)
            psd_df.to_csv(psd_file_path, index=False)

            print(f"Saved: {psd_file_path}")