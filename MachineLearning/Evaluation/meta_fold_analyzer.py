import os
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns

class MetaFoldAnalyzer:
    def __init__(self, base_path: str, model_name: str, set_name: str, subfolder: str):
        self.base_path = base_path
        self.model_name = model_name
        self.set_name = set_name
        self.subfolder = subfolder

        # Dynamisch Pfade setzen
        self.ml_results_path = os.path.join(base_path, "ML_Results", model_name, set_name, subfolder)
        self.metadata_path = os.path.join(base_path, "Metadata_analysis", model_name, set_name, subfolder)

        # Container für fold-spezifische Daten
        self.fold_errors_by_group = {}
        self.fold_class_distributions = {}
        self.fold_metrics = {}

    def load_all_folds(self):
        """
        Lädt alle relevanten Dateien (error_by_group, class_dist, metrics) aus den Ordnern.
        """
        # Suche alle Fold-Files (z. B. *_test_split_full_and_pred.csv)
        fold_files = glob.glob(os.path.join(self.ml_results_path, "*full_and_pred.csv"))
        for fold_file in fold_files:
            fold_name = os.path.basename(fold_file).replace("_full_and_pred.csv", "")

            # Lade Error by group
            err_path = os.path.join(self.metadata_path, f"{fold_name}_error_by_group.csv")
            if os.path.exists(err_path):
                self.fold_errors_by_group[fold_name] = pd.read_csv(err_path, index_col=0)

            # Lade Klassenverteilung
            dist_path = os.path.join(self.metadata_path, f"{fold_name}_class_dist_per_ResultID.csv")
            if os.path.exists(dist_path):
                self.fold_class_distributions[fold_name] = pd.read_csv(dist_path, header=[0,1], index_col=0)

            # Lade Metriken
            metrics_path = os.path.join(self.ml_results_path, f"{fold_name}_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    self.fold_metrics[fold_name] = json.load(f)

    def aggregate_error_by_group(self):
        """
        Gibt eine kombinierte Tabelle aus, in der die Fehlerrate pro Gruppe (z. B. ResultID)
        fold-übergreifend zusammengefasst wird.
        """
        combined = []
        for fold_name, df in self.fold_errors_by_group.items():
            df = df.copy()
            df["fold"] = fold_name
            df["group"] = df.index
            combined.append(df)

        if not combined:
            return pd.DataFrame()

        return pd.concat(combined, ignore_index=True)

    def analyze_class_imbalance_vs_accuracy(self):
        """
        Stellt Accuracy der Folds der durchschnittlichen Klassenverteilung gegenüber.
        """
        rows = []
        for fold_name, dist in self.fold_class_distributions.items():
            if fold_name in self.fold_metrics:
                acc = self.fold_metrics[fold_name].get("accuracy", None)
                rel = dist["rel"].mean().to_dict()
                rel["accuracy"] = acc
                rel["fold"] = fold_name
                rows.append(rel)

        return pd.DataFrame(rows)

    def plot_foldwise_error_heatmap(self, group_col_name="group"):
        """
        Zeigt eine Heatmap der Fehlerrate pro Fold x Gruppe (z. B. ResultID).
        """
        agg = self.aggregate_error_by_group()
        if agg.empty:
            print("Keine Fehlerdaten vorhanden.")
            return None

        pivot = agg.pivot(index="fold", columns=group_col_name, values="error_rate")
        fig, ax = plt.subplots(figsize=(min(18, pivot.shape[1]*0.7), 6))
        sns.heatmap(pivot, annot=False, cmap="Reds", ax=ax)
        ax.set_title("Fehlerrate pro Fold und Gruppe")
        plt.tight_layout()
        return fig
