import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_bis_sr_epochs(epochs_df_, output_csv, base_folder=r"D:\Daten\Initial_data\vitaldb_csvprocessed_BIS_BIS_SR"):
    """
    Processes BIS/SR values for given epochs and generates summary statistics.

    :param epochs_df_: DataFrame with columns ["Start", "End", "ResultID"].
    :param output_csv: Path to save the result CSV.
    :param base_folder: Folder containing <ResultID>.csv files
    :return: DataFrame with results.
    """
    results = []

    for _, row in epochs_df_.iterrows():
        start, end, result_id = row["Start"], row["End"], row["ResultID"]
        file_path = os.path.join(base_folder, f"{int(result_id)}.csv")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Load only needed columns
        df = pd.read_csv(file_path, usecols=["Time", "BIS/SR"])

        # Filter time range
        mask = (df["Time"] >= start) & (df["Time"] <= end)
        df_epoch = df.loc[mask].copy()

        if df_epoch.empty:
            print(f"No data for ResultID {result_id} between {start}-{end}")
            continue

        # Treat NaN as 0
        values = df_epoch["BIS/SR"].fillna(0).to_numpy()

        mean_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)

        results.append({
            "ResultID": result_id,
            "Start": start,
            "End": end,
            "BIS_SR_mean": mean_val,
            "BIS_SR_min": min_val,
            "BIS_SR_max": max_val
        })

    results_df_ = pd.DataFrame(results)

    # Save CSV
    results_df_.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    # --- Plot ---
    if not results_df_.empty:
        plt.figure(figsize=(10, 6))

        # Histogram of mean values
        plt.hist(results_df_["BIS_SR_mean"], bins=30, alpha=0.7, edgecolor="black")

        plt.xlabel("BIS/SR Mean Value")
        plt.ylabel("Count")
        plt.title("Distribution of BIS/SR Means Across Epochs")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig("D:\\Daten\\Other\\BIS_SR_data\\bis_sr_epochs_results.png")
        plt.show()

    return results_df_

def analyze_last_bis_activity(folder_path, output_csv):
    """
    Analyze BIS/SR activity in the last 600 seconds of all CSVs in a folder.

    :param folder_path: Path to folder containing <ResultID>.csv files
    :param output_csv: Path to save results as CSV
    """
    results = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            result_id = os.path.splitext(file)[0]
            file_path = os.path.join(folder_path, file)

            try:
                df = pd.read_csv(file_path)

                if "Time" not in df.columns or "BIS/SR" not in df.columns:
                    print(f"Skipping {file}: Missing required columns.")
                    continue

                # Last 600 seconds = 10 minutes
                max_time = df["Time"].max()
                df_last = df[df["Time"] >= max_time - 600].copy()

                # treat NaN values as 0
                df_last["BIS/SR"] = df_last["BIS/SR"].fillna(0)

                # Position of last valid value
                valid_rows = df_last[df_last["BIS/SR"] > 0]

                if not valid_rows.empty:
                    last_time = valid_rows["Time"].max()
                    seconds_before_end = max_time - last_time
                else:
                    seconds_before_end = 600  # no valid value found

                results.append({"ResultID": result_id,
                                "Seconds_before_end": seconds_before_end})

            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Save results
    results_df_ = pd.DataFrame(results)
    results_df_.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    return results_df_


if __name__ == "__main__":
    folder_path = r"D:\Daten\Initial_data\vitaldb_csvprocessed_BIS_BIS_SR"
    output_csv = r"D:\Daten\Other\BIS_SR_data\bis_sr_epochs_last_activity_all_patients.csv"
    analyze_last_bis_activity(folder_path, output_csv)

    # epochs_df = pd.read_csv("D:\\Daten\\Test_and_train\\Feature_sets\\Feature_sets_70_080_20_5\\Summary_Episodes_20_000.csv",)
    # results_df = process_bis_sr_epochs(epochs_df, output_csv="D:\\Daten\\Other\\BIS_SR_data\\bis_sr_epochs_results.csv",
    #                                    base_folder=r"D:\Daten\Initial_data\vitaldb_csvprocessed_BIS_BIS_SR")