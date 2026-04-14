from typing import Tuple, List
from pathlib import Path


class PathUtils:
    @staticmethod
    def return_A_B_C_D_path(prefix, parameters: dict) -> Path:
        """
        Calculates and returns a path variable defined by the class attributes,
        refractory time, min_episode_length, mac_threshold, and bis_threshold.
        Example: result_70_080_20_5

        :param prefix: Prefix of the subfolder name variable.
        :param parameters: A dictionary with all mentioned parameters

        :return: Subfolder path variable.
        """
        # Extract parameters from dict
        mac_threshold = parameters["mac_threshold"]
        bis_threshold = parameters["bis_threshold"]
        min_episode_length = parameters["min_episode_length"]
        refractory_time = parameters["refractory_time"]

        # leave 2 digits after decimal point and remove it afterward: 0.5 -> 050, 0.25 -> 025 etc.
        mac_threshold = f"{mac_threshold:.2f}".replace(".", "")
        return Path(f"{prefix}_{bis_threshold}_{mac_threshold}_{min_episode_length}_{refractory_time}")

    @staticmethod
    def return_X_Y_name(parameters: dict) -> Path:
        """
        Calculates and returns a path variable defined by the class attributes:
        merged_episodes, fixed_window_size, and overlap.
        Example: Summary_Episode_20_000

        :param parameters: A dictionary with all mentioned parameters
        :return: Subfolder path variable.
        """
        # extract parameters from dict
        merged_episodes = parameters["merged_episodes"]
        fixed_window_size = parameters["fixed_window_size"]
        overlap = parameters["overlap"]

        # decide on Episodes name
        episode_name = "Summary_Episodes"
        if merged_episodes:
            episode_name = "Summary_Merged_Episodes"
        # leave 2 digits after the decimal point and remove it afterward: 0.5 -> 050, 0.25 -> 025 etc.
        overlap = f"{overlap:.2f}".replace(".", "")
        return Path(f"{episode_name}_{fixed_window_size}_{overlap}")

    @staticmethod
    def list_files_in_folder(folder_path: Path, extension_filter: str = None, print_to_console=False, fullpaths=False)\
            -> Tuple[List[Path], int]:
        """
        Lists all files in a folder, optionally filtered by file extension.

        :param folder_path: Path to the folder to list files from.
        :param extension_filter: Optional file extension to filter by (e.g. ".csv" or ".txt").
        :param print_to_console: If true will print out all files to console.
        :param fullpaths: If true returns full file paths instead of just file names.
        :returns: List of file names and the total count of these files (list of file, length of list).
        """
        if not folder_path.exists():
            print(f"Folder does not exist: {folder_path}")
            return [], 0

        all_files = [f for f in folder_path.iterdir() if f.is_file()] # Aggregate only files

        if extension_filter:
            all_files = [f for f in all_files if f.suffix.lower() == extension_filter.lower()]

        if not fullpaths:
            all_files = [Path(f.name) for f in all_files]

        if print_to_console:
            for file in all_files:
                print(file)

        # Print information based on the extension filter flag
        print(f"\nTotal files{f' with extension {extension_filter}' if extension_filter else ''}: {len(all_files)}")

        return all_files, len(all_files)

    @staticmethod
    def clear_folder(folder_path: Path):
        """Deletes all files in a folder of given path, not including subfolders."""

        # Nothing to do if already deleted
        if not folder_path.exists():
            print(f"Folder does not exist: {folder_path}")
            return

        for filename in folder_path.iterdir():
            if filename.is_file():  # Delete only files, not subdirs
                try:
                    filename.unlink()
                    print(f"Deleted file: {filename}")
                except Exception as e:
                    print(f"Error deleting file {filename}: {e}")
