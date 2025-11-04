from MachineLearning.IO.load_data import LoadData,PathUtils
from MachineLearning.Utils.plots import Plots
import pandas as pd
Loader = LoadData()
Plotter = Plots()

patient_id = 1127

def plot_BIS_and_MAC(patient_id: int):
    # Assemble Folderpath
    folderpath = Loader.return_folder_path(["initial_data", "combined_raw_data"])
    filename = f"{patient_id}.csv"
    csv_path = PathUtils.return_anypath(folderpath, filename)

    df = pd.read_csv(csv_path)
    cols = ["BIS_BIS", "Primus_MAC"]
    Plotter.plot_signals_over_time(df, cols, f"BIS and MAC for patient {patient_id}")

if __name__ == "__main__":
    plot_BIS_and_MAC(patient_id)