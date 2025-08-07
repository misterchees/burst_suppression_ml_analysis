from MachineLearning.Models.pca_analyzer import PCAAnalyzer
from MachineLearning.IO.load_data import LoadData
import pandas as pd

hyperparams = {
    "merged_episodes": False,
    "bis_threshold": 70,
    "mac_threshold": 0.8,
    "min_episode_length": 20,
    "refractory_time": 5,
    "fixed_window_size": 20,
    "overlap": 0.0
}

loader = LoadData()
class_1_df, class_0_df = loader.load_combined_features_df(hyperparams, "awake", "faw")
df = pd.concat([class_0_df, class_1_df], ignore_index=True)
labels = df["label"] if "label" in df else None

analyzer = PCAAnalyzer(n_components=5)
pca_result = analyzer.fit_transform(df)

# 2D-Plot
analyzer.plot_components(labels=labels)

# Scree Plot
analyzer.plot_scree()

# Feature contributions to PC1
top_features = analyzer.get_feature_contributions(pc_index=0, top_n=10)
print(top_features)
