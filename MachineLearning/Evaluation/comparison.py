import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Set

from matplotlib import pyplot as plt
from MachineLearning.IO.load_data import LoadData
from MachineLearning.Features.transforms import Transforms
from MachineLearning.Utils.plots import Plots


class Comparison:

    def __init__(self):
        pass

    @staticmethod
    def compare_filtered_and_unfiltered_eeg(result_id: int, channel=1, log_scale=True, y_scale="raw",
                                            same_plot=True):
        """
        Will retrieve the raw and the filtered EEG for given Patient ID and Plot both PSDs to compare filtering.
        :param result_id: Patient ID
        :param channel: Channel of EEG (Options: 1 or 2)
        :param log_scale: Boolean to enable/disable log scaling in plots
        :param y_scale: Value to keep both y_axis in the exact same range.
        Options are 'filtered' for y_axis of filtered EEG plot, 'raw' for y_axis of raw EEG plot and 'min-max' for
        :param same_plot: Boolean to enable/disable plotting both curves into the same plot. Ignores y_scale.
        farthest limits of both EEGs.
        """
        loader = LoadData()
        transforms = Transforms(tuple("faw"), {})
        # Load EEGs from same Patient ID to compare
        fs_raw, raw_eegs = loader.return_eeg_tuple(result_id, False)
        fs_filt, filtered_eegs = loader.return_eeg_tuple(result_id, True)

        # Extract EEG of given channel
        raw_eeg = raw_eegs[:, channel - 1]
        filtered_eeg = filtered_eegs[:, channel - 1]

        # Create PSD for both
        raw_freq, raw_power = transforms.return_psd(raw_eeg, fs_raw)
        filt_freq, filt_power = transforms.return_psd(filtered_eeg, fs_filt)

        if same_plot:
            # Create one plot and write second plot into it with different color and label
            fig, ax = Plots.plot_psd(None, raw_freq, raw_power, "raw EEG", log_scale=log_scale)
            # Retrieve y-scale of raw ax scale to use as global ax for both
            if y_scale == "raw":
                raw_ax_y = ax.get_ylim()

            Plots.plot_psd((fig, ax), filt_freq, filt_power, "filtered EEG", "green",
                           f"Power Spectral Density of Patient {result_id}", log_scale)
            ############ Change this later. For now assuming raw y-scale is smaller im scaling it down to it
            if y_scale == "raw":
                ax.set_ylim(raw_ax_y)
        else:
            # Make one Plot with two subplots
            fig, axes = Plots.create_subplot_grid(2)
            raw_ax = axes[0]
            filt_ax = axes[1]
            Plots.plot_psd((fig, raw_ax), raw_freq, raw_power,
                           f"raw_EEG Power Spectral Density of Patient{result_id}", log_scale=log_scale)
            Plots.plot_psd((fig, filt_ax), filt_freq, filt_power,
                           f"filtered_EEG Power Spectral Density of Patient{result_id}", log_scale=log_scale)

            if y_scale == "min-max":
                Plots.align_axis(raw_ax, filt_ax, min_max_scale=True)
            elif y_scale == "raw":
                Plots.align_axis(raw_ax, filt_ax)
            elif y_scale == "filtered":
                Plots.align_axis(filt_ax, raw_ax)
            else:
                raise ValueError(f"Value '{y_scale}' for y_scale doesn't exist. "
                                 f"Valid options are: 'raw', 'filtered' or 'min-max'")

        plt.show()

    @staticmethod
    def compare_csv_to_psd_folder(
            csv_data: str | Path | pd.DataFrame,
            psd_folder: str | Path,
            key_cols: Tuple[str, str, str] = ("Start", "End", "ResultID"),
            return_missing: bool = False
    ) -> Dict[str, object]:
        """
        Compares the composite keys (Start, End, ResultID) of one CSV file
        against the keys encoded in PSD filenames inside *psd_folder*.

        PSD‑Filenames must follow: 'PSD_<start>_<end>_<result_id>.csv'

        :param csv_data: Path to the comparison CSV or the comparison df itself.
        :param psd_folder: Folder containing many PSD_*.csv files.
        :param key_cols: Column order that forms the key in the CSV.
        :param return_missing: If *True*, also return sets of missing keys.
        :returns: Dict with subset flags and (optionally) missing key sets.
        """
        # --read only the relevant columns from the CSV -----------------------
        df = Comparison._return_key_cols(csv_data, key_cols)

        # to ensure identical matching, cast to str and build hashable tuples
        keys_csv = set(
                        tuple(str(row[col]) for col in key_cols)
                        for _, row in df.iterrows()
        )

        # --extract keys from PSD filenames -----------------------------------
        pattern = re.compile(
            r"^PSD_(?P<start>[^_]+)_(?P<end>[^_]+)_(?P<rid>[^_]+)\.csv$", re.IGNORECASE
        )

        keys_psd: Set[Tuple[str, str, str]] = set()
        for fname in os.listdir(psd_folder):
            match = pattern.match(fname)
            if match:
                tup = (
                    match.group("start"),
                    match.group("end"),
                    match.group("rid"),
                )
                keys_psd.add(tup)

        # --subset relations ---------------------------------------------------
        a_in_b = keys_csv.issubset(keys_psd)
        b_in_a = keys_psd.issubset(keys_csv)

        result: Dict[str, object] = {
            "a_in_b": a_in_b,  # alle CSV‑Keys in PSD‑Filenames?
            "b_in_a": b_in_a,  # alle PSD‑Keys auch im CSV?
        }

        if return_missing:
            result["missing_from_b"] = keys_csv - keys_psd  # im CSV, aber nicht im PSD‑Ordner
            result["missing_from_a"] = keys_psd - keys_csv  # im PSD‑Ordner, aber nicht im CSV

        return result

    @staticmethod
    def compare_two_csv(
            csv_data_a: str | Path | pd.DataFrame,
            csv_data_b: str | Path | pd.DataFrame,
            key_cols: Tuple[str, str, str] = ("Start", "End", "ResultID"),
            return_missing: bool = False
    ) -> Dict[str, object]:
        """
        Checks whether two CSV files contain identical or nested key‑sets defined
        by the columns in *key_cols*.

        :param csv_data_a: Path to the first CSV or the df itself.
        :param csv_data_b: Path to the second CSV or the df itself.
        :param key_cols: Tuple with the column names that form the composite key.
        :param return_missing: If *True*, also return the concrete rows missing on
                               either side (as sets of tuples).
        :returns: Dict with boolean subset flags and, if requested, the missing
                  key tuples.
                  Example::
                      {
                          "a_in_b": True,
                          "b_in_a": False,
                          "missing_from_a": {...},  # only if return_missing=True
                          "missing_from_b": {...}
                      }
        """
        # --- read only the needed columns -------------------
        df_a = Comparison._return_key_cols(csv_data_a, key_cols)
        df_b = Comparison._return_key_cols(csv_data_b, key_cols)

        # --- build hashable key‑sets ------------------------
        keys_a: Set[Tuple] = set(map(tuple, df_a[key_cols].to_records(index=False)))
        keys_b: Set[Tuple] = set(map(tuple, df_b[key_cols].to_records(index=False)))

        # --- subset checks ---------------------------------
        a_in_b = keys_a.issubset(keys_b)
        b_in_a = keys_b.issubset(keys_a)

        result: Dict[str, object] = {
            "a_in_b": a_in_b,
            "b_in_a": b_in_a,
        }

        if return_missing:
            result["missing_from_b"] = keys_a - keys_b  # rows in A, not in B
            result["missing_from_a"] = keys_b - keys_a  # rows in B, not in A

        return result

    @staticmethod
    def _return_key_cols(csv_data: str | Path | pd.DataFrame, key_cols: Tuple[str, str, str]) -> pd.DataFrame:
        """
        Helping function to unpack the relevant columns from csv data.
        :param csv_data: Path to the comparison CSV or the comparison df itself.
        :param key_cols: Relevant columns.
        :return: Pandas Dataframe with relevant columns.
        """
        if isinstance(csv_data, (str, Path)):
            df = pd.read_csv(csv_data, usecols=list(key_cols))
        elif isinstance(csv_data, pd.DataFrame):
            df = csv_data[list(key_cols)]
        else:
            raise TypeError("csv data must be a Path/str or a pandas DataFrame")

        return df
