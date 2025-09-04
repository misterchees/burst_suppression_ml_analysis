"""
Here is the place to test any code.
"""
import pandas as pd
import numpy as np
from MachineLearning.Evaluation.split_manager import SplitManager

# Beispiel-DataFrames simulieren
class_1_df = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [10, 20, 30],
    "Start":[1,2,3],
    "End": [10, 20, 30],
    "ResultID": [1, 2, 3]
})
class_0_df = pd.DataFrame({
    "A": [4, 5, 6, 7],
    "B": [40, 50, 60, 70],
    "Start":[4,5,6,7],
    "End": [50, 40, 60,80],
    "ResultID": [7, 8, 9,10]
})

split_manager = SplitManager({}, "faw", "awake")
split_manager.class_0_df = class_0_df
split_manager.class_1_df = class_1_df

# Normalisierung, die zu checken ist
split_manager.normalize_data()

normalized_class_1 = split_manager.class_1_df
normalized_class_0 = split_manager.class_0_df
# --- Tests ---
# 1. Anzahl Zeilen muss identisch sein
assert len(class_1_df) == len(normalized_class_1), "class_1 length mismatch"
assert len(class_0_df) == len(normalized_class_0), "class_0 length mismatch"

# 2. Spalten müssen gleich sein
assert list(class_1_df.columns) == list(normalized_class_1.columns), "class_1 columns mismatch"
assert list(class_0_df.columns) == list(normalized_class_0.columns), "class_0 columns mismatch"

# 3. Reihenfolge muss gleich bleiben → wir testen mit Index
for i in range(len(class_1_df)):
    assert class_1_df.index[i] == i, "Index mismatch in class_1 after reset_index"

for i in range(len(class_0_df)):
    assert class_0_df.index[i] == i, "Index mismatch in class_0 after reset_index"

# 4. Als sanity check: Normalisierte Werte sollten ≠ Originalwerte sein
assert not np.allclose(class_1_df.values, normalized_class_1.values), "Normalization didn't change class_1"
assert not np.allclose(class_0_df.values, normalized_class_0.values), "Normalization didn't change class_0"

print("✅ Struktur und Normalisierung sehen korrekt aus!")

