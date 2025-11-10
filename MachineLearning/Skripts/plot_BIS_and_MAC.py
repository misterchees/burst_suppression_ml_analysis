from MachineLearning.IO.load_data import LoadData, PathUtils
from MachineLearning.Utils.plots import Plots
import pandas as pd

loader = LoadData()
plotter = Plots()

_patient_id = 1127


def plot_BIS_and_MAC(patient_id: int, eeg: bool):
    # Assemble Folderpath
    folderpath = loader.return_folder_path(["initial_data", "combined_raw_data"])
    filename = f"{patient_id}.csv"
    csv_path = PathUtils.return_anypath(folderpath, filename)

    df = pd.read_csv(csv_path)
    if not eeg:
        cols = ["BIS_BIS", "Primus_MAC"]
        plotter.plot_signals_over_time(df, cols, f"BIS and MAC for patient {patient_id}")
    else:
        filtered = False # No need for filtered data here
        fs, all_channel_eeg = loader.load_eeg_data(patient_id, filtered)
        one_channel_eeg = all_channel_eeg[:, 0]  # channel 1 of EEG
        plotter.plot_eeg_bis_mac_over_time(one_channel_eeg, fs, df, filtered, ["BIS_BIS"],"Primus_MAC" )


if __name__ == "__main__":
    plot_BIS_and_MAC(_patient_id, True)
