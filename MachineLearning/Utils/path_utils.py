import os


class PathUtils:
    @staticmethod
    def return_anypath(*path_parts: str) -> str:
        """
        Creates any path build from given path_parts.
        :param path_parts: Any number of path parts in the right order to assemble them as strings.
        :return: A path built from given path_parts as a string.
        """
        return str(os.path.join(*path_parts))

    @staticmethod
    def return_A_B_C_D_name(prefix, parameters: dict) -> str:
        """
        Calculates and returns a name variable defined by the class attributes,
        refractory time, min_episode_length, mac_threshold and bis_threshold.
        Example: result_70_080_20_5

        :param prefix: Prefix of the subfolder name variable.
        :param parameters: A dictionary with all mentioned parameters

        :return: Subfolder string variable.
        """
        # extract parameters from dict
        mac_threshold = parameters["mac_threshold"]
        bis_threshold = parameters["bis_threshold"]
        min_episode_length = parameters["min_episode_length"]
        refractory_time = parameters["refractory_time"]

        # leave 2 digits after decimal point and remove it afterward: 0.5 -> 050, 0.25 -> 025 etc.
        mac_threshold = f"{mac_threshold:.2f}".replace(".", "")
        return f"{prefix}_{bis_threshold}_{mac_threshold}_{min_episode_length}_{refractory_time}"

    @staticmethod
    def return_X_Y_name(parameters: dict) -> str:
        """
        Calculates and returns a name variable defined by the class attributes:
        merged_episodes, fixed_window_size and overlap.
        Example: Summary_Episode_20_000

        :param parameters: A dictionary with all mentioned parameters
        :return: Subfolder string variable.
        """
        # extract parameters from dict
        merged_episodes = parameters["merged_episodes"]
        fixed_window_size = parameters["fixed_window_size"]
        overlap = parameters["overlap"]

        # decide on Episodes name
        episode_name = "Summary_Episodes"
        if merged_episodes:
            episode_name = "Summary_Merged_Episodes"
        # leave 2 digits after decimal point and remove it afterward: 0.5 -> 050, 0.25 -> 025 etc.
        overlap = f"{overlap:.2f}".replace(".", "")
        return f"{episode_name}_{fixed_window_size}_{overlap}"

    @staticmethod
    def return_A_B_C_D_X_Y_path(prefix: str, parameters: dict) -> str:
        """
        Returns a path-like string. For details look into return_A_B_C_D_name and return_X_Y_name functions.
        :param prefix: Prefix of abcd folder
        :param parameters: parameters that decide about the variables.
        :return: <prefix>_A_B_C_D\\<Episode>_X_Y
        """
        abcd_folder = PathUtils.return_A_B_C_D_name(prefix, parameters)
        xy_folder = PathUtils.return_X_Y_name(parameters)
        return str(os.path.join(abcd_folder, xy_folder))

    @staticmethod
    def return_filename_from_fullpath(filepath: str, extension=False) -> str:
        """
        Returns the file name from the given file path.

        This method extracts the base name of a file from its file path. By default,
        it removes the file extension from the returned name, but the full file name
        including its extension can be returned if explicitly specified.

        :param filepath: The full path of the file from which the name should
            be extracted.
        :type filepath: str
        :param extension: A flag indicating whether the file extension should be
            included in the returned file name. Defaults to False.
        :type extension: bool, optional
        :return: The file name extracted from the file path. By default, it excludes
            the file extension. If `extension` is True, the file name with its
            extension is returned instead.
        :rtype: str
        """
        filename = os.path.basename(filepath)  # foo/bar/baz.txt -> baz.txt

        if extension:
            return filename
        else:
            return os.path.splitext(filename)[0]  # baz.txt -> baz


    @staticmethod
    def return_node_name(parameters: dict, node_type: str) -> str:
        epoch_length = parameters["fixed_window_size"]
        if node_type == "awake":
            return f"Awake_{epoch_length}"
        elif node_type == "normal_an":
            return f"Normal_ane_{epoch_length}"
        raise ValueError(f"Unknown node type {node_type}. Valid types are 'awake' and 'normal_an'")

    @staticmethod
    def filepath_exists(filepath: str) -> bool:
        return os.path.isfile(filepath)

    @staticmethod
    def list_files_in_folder(folder_path: str, extension_filter:str = None, print_to_console=False):
        """
        Lists all files in a folder, optionally filtered by file extension.

        :param folder_path: Path to the folder to list files from.
        :param extension_filter: Optional file extension to filter by (e.g. ".csv" or ".txt").
        :param print_to_console: If true will print out all files to console.
        :returns: List of file names and the total count of these files.
        """
        if not os.path.exists(folder_path):
            print(f"Folder does not exist: {folder_path}")
            return [], 0

        all_files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]

        if extension_filter:
            all_files = [f for f in all_files if f.lower().endswith(extension_filter.lower())]

        if print_to_console:
            for file in all_files:
                print(file)

        print(f"\nTotal files{f' with extension {extension_filter}' if extension_filter else ''}: {len(all_files)}")

        return all_files, len(all_files)

    @staticmethod
    def clear_folder(folder_path: str):
        """Deletes all files in a folder of given path, not including subfolders."""

        # Nothing to do if already deleted
        if not os.path.exists(folder_path):
            print(f"Folder does not exist: {folder_path}")
            return

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):  # Deletes only files, not subdirs
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

    @staticmethod
    def assemble_psd_file_name(start: int, end: int, result_id: int) -> str:
        """Assembles the name of a PSD file with given metadata."""
        if start is None or end is None or result_id is None:
            raise ValueError("start, end and result_id must be values")

        return f"PSD_{start}_{end}_{result_id}.csv"

    @staticmethod
    def diff_epochs_vs_psd_files(psd_file_path, epochs) -> bool:
        # Get csv files and number of files in psd_path
        psd_files_list, existing_psd_count = PathUtils.list_files_in_folder(psd_file_path, ".csv")

        # If there aren't as many epochs as psd_files, there is a diff -> return true
        if existing_psd_count != len(epochs):
            return True
        # If not all epochs have a calculated psd counterpart, there is a diff -> return true
        else:
            psd_file_set = set(psd_files_list) # convert to set for optimal search efficiency
            # Assemble each PSD filename from epoch metadata and check if exists. If not exist -> return true
            for epoch in epochs:
                start, end, result_id, _, _ = epoch # Get relevant data of epochs
                psd_file_name = PathUtils.assemble_psd_file_name(start, end, result_id)
                if psd_file_name not in psd_file_set:
                    return True
        # Number of epochs is equal to number of PSD files and every epoch has its PSD counterpart
        return False

