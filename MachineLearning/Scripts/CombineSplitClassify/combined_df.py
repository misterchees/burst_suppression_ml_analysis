import pandas as pd

def combine_faw_aw(faw_path, aw_path, output_path):
    """
    Combine FAW and AW CSV files into a single CSV with class labels.

    :param faw_path: Path to FAW CSV file
    :param aw_path: Path to AW CSV file
    :param output_path: Path where combined CSV should be saved
    """
    # Read both CSVs
    df_faw = pd.read_csv(faw_path)
    df_aw = pd.read_csv(aw_path)

    # Add class labels
    df_faw["class"] = "FAW"
    df_aw["class"] = "AW"

    # Combine
    combined_df = pd.concat([df_faw, df_aw], ignore_index=True)

    # Round all numeric columns except Start, End, ResultID
    cols_to_round = [c for c in combined_df.columns if c not in ["Start", "End", "ResultID", "class"]]
    combined_df[cols_to_round] = combined_df[cols_to_round].round(2)

    # Save to CSV
    combined_df.to_csv(output_path, index=False)

    return combined_df

if __name__ == "__main__":
    df_faw_path = r"D:\Daten\Test_and_train\Feature_sets\Awake_20.csv"
    df_aw_path = r"D:\Daten\Test_and_train\Feature_sets\Feature_sets_70_080_20_5\Summary_Episodes_20_000.csv"
    output_df_path = r"D:\Daten\comb_df.csv"
    combine_faw_aw(df_faw_path, df_aw_path, output_df_path)
