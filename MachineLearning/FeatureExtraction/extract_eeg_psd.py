import os
import warnings
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import welch
import matplotlib.pyplot as plt


def extract_eeg_psd(preprocessing_dir, vitaldb_dir, output_dir, channel=1, nperseg_seconds=2):
    """
    Berechnet PSDs für EEG-Ausschnitte basierend auf Summary_Episodes CSV-Dateien.

    Parameters:
    - preprocessing_dir: Pfad zu "preprocessing" (Ordner mit result_A_B_C_D-Unterordnern)
    - vitaldb_dir: Pfad zu "vitalDB_mat_EEG" (enthält X.mat Dateien)
    - output_dir: Pfad zu "Features" (hier werden PSDs gespeichert)
    - channel: EEG-Channel (1 oder 2)
    - nperseg_seconds: Fensterlänge für Welch in Sekunden
    """
    for subfolder in os.listdir(preprocessing_dir):
        if subfolder.startswith("result_"):
            # A_B_C_D extrahieren
            subfolder_parts = subfolder.split('_')[1:]  # ['A', 'B', 'C', 'D']
            result_csv_path = os.path.join(preprocessing_dir, subfolder, "Summary_Episodes.csv") # Pfad zur Datei

            if not os.path.isfile(result_csv_path):
                warnings.warn(f"CSV nicht gefunden: {result_csv_path}")
                continue

            df = pd.read_csv(result_csv_path)

            # Output-Ordner anlegen: PSD_A_B_C_D
            psd_subfolder = f"PSD_{'_'.join(subfolder_parts)}"
            psd_output_path = os.path.join(output_dir, psd_subfolder)
            os.makedirs(psd_output_path, exist_ok=True)

            for _, row in df.iterrows():
                result_id = int(row['ResultID'])
                start_time = int(row['Start'])
                end_time = int(row['End'])

                mat_file_path = os.path.join(vitaldb_dir, f"{result_id}.mat")
                if not os.path.isfile(mat_file_path):
                    warnings.warn(f"MAT-Datei nicht gefunden: {mat_file_path}")
                    continue

                # .mat Datei laden
                mat_data = scipy.io.loadmat(mat_file_path)
                fs = int(mat_data['fs'].squeeze())
                rawEEG = mat_data['rawEEG']

                # Prüfen ob Channel existiert
                if channel not in [1, 2]:
                    raise ValueError(f"Channel muss 1 oder 2 sein. Gewählt: {channel}")

                eeg_signal = rawEEG[:, channel - 1]

                # Zeitbereich in Samples
                start_sample = int(start_time * fs)
                end_sample = int(end_time * fs)
                eeg_segment = eeg_signal[start_sample:end_sample]

                # Welch PSD berechnen
                nperseg = int(nperseg_seconds * fs)
                frequencies, psd = welch(eeg_segment, fs=fs, nperseg=nperseg)

                # Ergebnisse als DataFrame
                psd_df = pd.DataFrame({
                    'Frequency_Hz': frequencies,
                    'PSD_V2_per_Hz': psd
                })

                # Speichern als PSD_H_K_L.csv
                psd_filename = f"PSD_{result_id}_{start_time:.2f}_{end_time:.2f}.csv"
                psd_file_path = os.path.join(psd_output_path, psd_filename)
                psd_df.to_csv(psd_file_path, index=False)

                print(f"Gespeichert: {psd_file_path}")
