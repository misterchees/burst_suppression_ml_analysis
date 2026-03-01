import re
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Set

from matplotlib import pyplot as plt
from MachineLearning.IO.load_data import LoadData
from MachineLearning.Features.transforms import Transforms
from MachineLearning.Utils.plots import Plots
from MachineLearning.Utils.path_manager import PathManager


class Comparison:

    def __init__(self, pm:PathManager):
        self.pm = pm
        self.loader = LoadData(self.pm)

    def compare_filtered_and_unfiltered_eeg(self, result_id: int, channel=1, log_scale=True, y_scale="raw",
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
        transforms = Transforms(tuple("faw"), {}, {})
        # Load EEGs from same Patient ID to compare
        fs_raw, raw_eegs = self.loader.load_eeg_data(result_id, False)
        fs_filt, filtered_eegs = self.loader.load_eeg_data(result_id, True)

        # Extract EEG of given channel
        raw_eeg = raw_eegs[:, channel - 1]
        filtered_eeg = filtered_eegs[:, channel - 1]

        # Create PSD for both
        nperseg_seconds = 2
        raw_freq, raw_power = transforms.calculate_psd_welch(raw_eeg, fs_raw, nperseg_seconds)
        filt_freq, filt_power = transforms.calculate_psd_welch(filtered_eeg, fs_filt, nperseg_seconds)

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

        psd_folder = Path(psd_folder)
        if psd_folder.is_dir():  # If folder doesn't exist, there are no keys to add
            for fname in psd_folder.iterdir():
                match = pattern.match(str(fname))
                if match:
                    tup = (
                        match.group("start"),
                        match.group("end"),
                        match.group("rid"),
                    )
                    keys_psd.add(tup)

        result = Comparison._return_diff_dict_from_sets(keys_csv, keys_psd, return_missing)
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
                  Example:
                      {
                          "a_in_b": True,
                          "b_in_a": False,
                          "missing_from_a": {...}, # if return_missing=True
                          "missing_from_b": {...}  # if return_missing=True
                      }
        """
        # --- read only the needed columns -------------------
        df_a = Comparison._return_key_cols(csv_data_a, key_cols)
        df_b = Comparison._return_key_cols(csv_data_b, key_cols)

        # --- build hashable key‑sets ------------------------
        keys_a = set(
            tuple(str(row[col]) for col in key_cols)
            for _, row in df_a.iterrows()
        )

        keys_b = set(
            tuple(str(row[col]) for col in key_cols)
            for _, row in df_b.iterrows()
        )

        # Create and return a diff dictionary from key sets
        result = Comparison._return_diff_dict_from_sets(keys_a, keys_b, return_missing)
        return result

    @staticmethod
    def _return_key_cols(csv_data: str | Path | pd.DataFrame, key_cols: Tuple[str, str, str]) -> pd.DataFrame:
        """
        Helping function to unpack the relevant columns from csv data.
        :param csv_data: Path to the comparison CSV or the comparison df itself.
        :param key_cols: Relevant columns.
        :return: Pandas Dataframe with relevant columns.
        """
        # If csv_data is a path, check if the file exists and read. If not, create an empty df with empty columns
        if isinstance(csv_data, (str, Path)):
            csv_data = Path(csv_data)
            if csv_data.is_file():
                df = pd.read_csv(csv_data, usecols=list(key_cols))
            else:
                df = pd.DataFrame(columns=list(key_cols))

        elif isinstance(csv_data, pd.DataFrame):
            df = csv_data[list(key_cols)]
        else:
            raise TypeError("csv data must be a Path/str or a pandas DataFrame")

        return df

    @staticmethod
    def _return_diff_dict_from_sets(keys_a: set, keys_b: set, return_missing: bool) -> Dict[str, object]:
        """
        Compares two sets of keys and determines their subset relationships, as well
        as computes missing elements depending on the input parameters.

        :param keys_a: The first set of keys to compare.
        :type keys_a: set
        :param keys_b: The second set of keys to compare.
        :type keys_b: set
        :param return_missing: Flag indicating whether to return missing elements from
            either set in the result.
        :type return_missing: bool
        :return: A dictionary containing the subset relationships and,
            optionally, the missing elements if `return_missing` is True. The keys of
            the dictionary are:
                - "a_in_b": Boolean indicating if `keys_a` is a subset of `keys_b`.
                - "b_in_a": Boolean indicating if `keys_b` is a subset of `keys_a`.
                - "missing_from_b": Set of elements in `keys_a` but not in `keys_b`
                    (only if `return_missing` is True).
                - "missing_from_a": Set of elements in `keys_b` but not in `keys_a`
                    (only if `return_missing` is True).
        :rtype: Dict[str, object]
        """
        # subset checks to find diffs
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