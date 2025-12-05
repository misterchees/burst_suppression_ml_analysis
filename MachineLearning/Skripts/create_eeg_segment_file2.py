import pandas as pd
import scipy.io
import os

# --- CONFIGURATION ---
# Input Directories
PATH_EEG_DATA = r"E:\Daten\Initial_data\vitalDB_mat_EEG"
PATH_FILTERED_EEG_DATA = r"E:\Daten\trimmed_normalized_filtered_05_40"
PATH_FAW_DATA = r"E:\Daten\Other\EEG_segments_for_more_powerful_classification_models\Input data"
PATH_ANESTART = r"E:\Daten\anestart_analysis_results.csv"

# Output Directory
PATH_OUTPUT = r"E:\Daten\Other\EEG_segments_for_more_powerful_classification_models\Output data"

# Signal Parameters
SAMPLING_RATE = 128
SEGMENT_SECONDS = 7
SAMPLES_PER_SEGMENT = SEGMENT_SECONDS * SAMPLING_RATE  # 7 * 128 = 896
OVERLAP_PERCENT = 0.75
FILTERED = True

# Generate column names (eeg_0 to eeg_895)
EEG_COL_NAMES = [f'eeg_{i}' for i in range(SAMPLES_PER_SEGMENT)]


def load_eeg_channel_1(filepath):
    """
    Loads the .mat file and extracts Channel 1 from 'rawEEG'.

    Args:
        filepath (str): Path to the .mat file.

    Returns:
        np.array: 1D array of the first EEG channel, or None if file error.
    """
    raw_data = None

    if FILTERED:
        # --- PARQUET LOADING ---
        try:
            # Read Parquet into DataFrame
            df = pd.read_parquet(filepath)

            # Convert to numpy array to ensure [:, 0] indexing works
            # We assume the filtered data structure mirrors the raw data (Time x Channels)
            raw_data = df.values

        except FileNotFoundError:
            print(f"Warning: File not found: {filepath}")
            return None
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None
    else:
        # --- MAT LOADING ---
        try:
            mat = scipy.io.loadmat(filepath)
            # check if rawEEG exists
            if 'rawEEG' not in mat:
                print(f"Warning: Key 'rawEEG' not found in {filepath}")
                return None
            # Extract raw data. Shape is expected to be (N, 2)
            raw_data = mat['rawEEG']
        except FileNotFoundError:
            print(f"Warning: File not found: {filepath}")
            return None
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

    # --- EXTRACTION ---
    # Safety check if raw_data is actually valid
    if raw_data is None:
        return None

    try:
        # We need Channel 1. Assuming 0-based indexing, this is column 0.
        # This works now for both because we converted the DataFrame to .values above
        eeg_channel_1 = raw_data[:, 0].flatten()
        return eeg_channel_1
    except IndexError:
        print(f"Error: Data in {filepath} has unexpected shape {raw_data.shape}.")
        return None


def process_label_1(anestart_df):
    """
    Generates segments for Label 1 (Conscious state before anesthesia).
    Logic: From t=0 to t=anestart with 75% overlap.
    """
    print("Processing Label 1 data (Baseline)...")
    segments = []

    # Calculate step size for sliding window (25% stride)
    step_size = int(SAMPLES_PER_SEGMENT * (1 - OVERLAP_PERCENT))

    for _, row in anestart_df.iterrows():
        # Ensure caseid matches the filename format
        case_id = str(int(row['caseid']))
        anestart_time = row['anestart']

        if FILTERED:
            file_path = os.path.join(PATH_FILTERED_EEG_DATA, f"{case_id}.parquet")
        else:
            file_path = os.path.join(PATH_EEG_DATA, f"{case_id}.mat")

        eeg_signal = load_eeg_channel_1(file_path)
        if eeg_signal is None:
            continue

        # Define the limit (in samples)
        max_sample_index = int(anestart_time * SAMPLING_RATE)

        # Ensure we don't exceed actual signal length
        max_idx = min(max_sample_index, len(eeg_signal))

        # Sliding Window Loop
        current_idx = 0
        # While the end of the current segment is within bounds
        while current_idx + SAMPLES_PER_SEGMENT <= max_idx:
            # Define end point of segment
            end_idx = current_idx + SAMPLES_PER_SEGMENT
            # Extract segment
            segment = eeg_signal[current_idx: current_idx + SAMPLES_PER_SEGMENT]

            # Construct row: [patient_id, start, end, eeg_0...eeg_895, label]
            row_data = [case_id, current_idx, end_idx] + segment.tolist() + [1]
            segments.append(row_data)

            # Move window
            current_idx += step_size

    return segments


def process_label_0(faw_filename, faw_folder):
    """
    Generates segments for Label 0 based on FAW CSV files.
    Logic: Uses defined Start times from CSV.
    """
    print(f"Processing Label 0 file: {faw_filename}...")
    faw_path = os.path.join(faw_folder, faw_filename)
    if not os.path.exists(faw_path):
        print(f"Error: FAW file not found: {faw_path}")
        return []

    faw_df = pd.read_csv(faw_path)
    segments = []

    # Group by ResultID (Patient) to load each .mat file only once per CSV
    grouped = faw_df.groupby('ResultID')

    for case_id_raw, group in grouped:
        case_id = str(int(case_id_raw))
        if FILTERED:
            file_path = os.path.join(PATH_FILTERED_EEG_DATA, f"{case_id}.parquet")
        else:
            file_path = os.path.join(PATH_EEG_DATA, f"{case_id}.mat")

        eeg_signal = load_eeg_channel_1(file_path)

        if eeg_signal is None:
            continue

        for _, row in group.iterrows():
            start_time = row['Start']

            # Calculate indices
            start_idx = int(start_time * SAMPLING_RATE)
            end_idx = start_idx + SAMPLES_PER_SEGMENT

            # Boundary check
            if end_idx <= len(eeg_signal):
                segment = eeg_signal[start_idx: end_idx]

                # Construct row: [patient_id, start, end, eeg_0...eeg_895, label]
                row_data = [case_id, start_idx, end_idx] + segment.tolist() + [0]
                segments.append(row_data)

    return segments


def main():
    # 0. Ensure Output Directory exists
    if not os.path.exists(PATH_OUTPUT):
        print(f"Creating output directory: {PATH_OUTPUT}")
        os.makedirs(PATH_OUTPUT)

    # 1. Generate Label 1 Data (Common for all subgroups)
    # ------------------------------------------------
    if not os.path.exists(PATH_ANESTART):
        print("Error: Anestart file not found!")
        return

    anestart_df = pd.read_csv(PATH_ANESTART)
    label_1_list = process_label_1(anestart_df)

    # Create DataFrame for Label 1
    columns = ['patient_id', 'segment_start', 'segment_end'] + EEG_COL_NAMES + ['label']
    df_label_1 = pd.DataFrame(label_1_list, columns=columns)

    print(f"Label 1 processing complete. Generated {len(df_label_1)} segments.")

    # 2. Process Label 0 for each subgroup and merge
    # ------------------------------------------------
    faw_files = [
        "Summary_Episodes_faw_minlength_10_fixlength_7_overlap_075.csv",
        "Summary_Episodes_faw_minlength_15_fixlength_7_overlap_075.csv",
        "Summary_Episodes_faw_minlength_20_fixlength_7_overlap_075.csv"
    ]

    for f_name in faw_files:
        # Extract Label 0 segments
        label_0_list = process_label_0(f_name, PATH_FAW_DATA)
        df_label_0 = pd.DataFrame(label_0_list, columns=columns)

        # Merge Label 1 and Label 0
        df_combined = pd.concat([df_label_1, df_label_0], ignore_index=True)

        # Construct output filename
        if FILTERED:
            output_filename = f_name.replace(".csv", ".parquet").replace("Summary_Episodes_faw_", "normalized_filtered_")
        else:
            output_filename = f_name.replace(".csv", ".parquet").replace("Summary_Episodes_faw_", "")
        output_path = os.path.join(PATH_OUTPUT, output_filename)

        print(f"Saving {len(df_combined)} rows to: {output_filename}")

        # Save as Parquet
        df_combined.to_parquet(output_path, index=False)

    print("All tasks completed successfully.")


if __name__ == "__main__":
    main()