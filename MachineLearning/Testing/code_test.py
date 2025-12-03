"""
Here is the place to test any code.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- KONFIGURATION ---
OUTPUT_DIR = r"D:\Daten\Other\EEG_segments_for_more_powerful_classification_models\Output data"
# Wähle eine Datei zum Testen
TEST_FILE = "minlength_10_fixlength_7_overlap_075.parquet"


def check_overlap():
    file_path = os.path.join(OUTPUT_DIR, TEST_FILE)
    if not os.path.exists(file_path):
        print(f"Datei nicht gefunden: {file_path}")
        return

    print(f"Lade {TEST_FILE}...")
    df = pd.read_parquet(file_path)

    # Filtere nur Label 1 Daten (da wir diese selbst generiert haben mit garantiertem Overlap)
    # Und sortiere sicherheitshalber, falls durch Concats die Reihenfolge litt (sollte aber passen)
    df_subset = df[df['label'] == 1].reset_index(drop=True)

    if len(df_subset) < 2:
        print("Nicht genug Daten für einen Vergleich.")
        return

    # Wir nehmen Zeile 0 und Zeile 1 des gleichen Patienten
    row_0 = df_subset.iloc[400]
    row_1 = df_subset.iloc[401]

    # Sicherstellen, dass es derselbe Patient ist
    if row_0['patient_id'] != row_1['patient_id']:
        print("Warnung: Zeile 0 und 1 gehören zu unterschiedlichen Patienten. Suche nach Sequenz...")
        # (Hier könnte man eine Schleife bauen, aber für den Test reicht meist der Anfang)
        return

    # --- METHODE 1: Punktueller Check (Deine Frage) ---
    val_prev_224 = row_0['eeg_224']
    val_curr_0 = row_1['eeg_0']

    print("\n--- METHODE 1: Punktueller Vergleich ---")
    print(f"Wert in Segment 0 an Pos 224: {val_prev_224}")
    print(f"Wert in Segment 1 an Pos 0:   {val_curr_0}")

    if np.isclose(val_prev_224, val_curr_0):
        print("✅ MATCH: Der Start des neuen Segments entspricht dem Index 224 des alten.")
    else:
        print("❌ FEHLER: Werte stimmen nicht überein.")

    # --- METHODE 2: Bereichs-Vergleich (Vollständiger Overlap) ---
    # Wir vergleichen eeg_224 bis eeg_895 von Zeile 0
    # mit eeg_0 bis eeg_671 von Zeile 1

    # Slice extrahieren (als numpy arrays)
    # Zeile 0: ab Index 224 bis Ende
    overlap_segment_0 = row_0.loc['eeg_224':'eeg_895'].to_numpy(dtype=float)

    # Zeile 1: ab Anfang bis (Ende - 224)
    # Das Ende ist Index 671 (weil 896 - 224 = 672 Elemente)
    overlap_segment_1 = row_1.loc['eeg_0':'eeg_671'].to_numpy(dtype=float)

    print("\n--- METHODE 2: Vollständiger Array Vergleich ---")
    # Floating point vergleich mit kleiner Toleranz
    if np.allclose(overlap_segment_0, overlap_segment_1):
        print(f"✅ MATCH: Alle {len(overlap_segment_0)} überlappenden Datenpunkte sind identisch.")
    else:
        diff = np.abs(overlap_segment_0 - overlap_segment_1)
        print(f"❌ FEHLER: Durchschnittliche Abweichung: {np.mean(diff)}")

    # --- METHODE 3: Visueller Plot ---
    print("\n--- METHODE 3: Plot wird erstellt... ---")

    # Volles Segment 0
    y0 = row_0.loc['eeg_0':'eeg_895'].to_numpy(dtype=float)
    # Volles Segment 1
    y1 = row_1.loc['eeg_0':'eeg_895'].to_numpy(dtype=float)

    x0 = np.arange(0, 896)
    x1 = np.arange(0, 896) + 224  # Verschiebung auf der X-Achse

    plt.figure(figsize=(12, 6))
    plt.plot(x0, y0, label='Segment 0 (Original)', color='blue', alpha=0.7)
    plt.plot(x1, y1, label='Segment 1 (Verschoben um 224)', color='orange', alpha=0.7, linestyle='--')

    plt.title(f"Overlap Check: Patient {row_0['patient_id']}")
    plt.xlabel("Samples (Zeit)")
    plt.ylabel("EEG Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    check_overlap()