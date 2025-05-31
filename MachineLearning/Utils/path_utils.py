import os
import warnings


class PathUtils:
    @staticmethod
    def create_anypath(*path_parts: str) -> str:
        """
        Creates any path build from given path_parts.
        :param path_parts: Any number of path parts in the right order to assemble them.
        :return: A path build from given path_parts as string.
        """
        return str(os.path.join(*path_parts))

    @staticmethod
    def create_csv_fullpath(directory, prefix, parameters: dict) -> str:
        """
        Calculates and returns a fullpath string variable to the desired csv defined by the class parameters
        merged_episodes, refractory time, min_episode_length, mac_threshold, bis_threshold, overlap
        and fixed_window_size.
        Example: result_70_080_20_5\\Summary_Episodes_20_000.csv

        :param parameters: A dictionary with all mentioned parameters
        :param directory: Initial Directory for the fullpath.
        :param prefix: Prefix of the subfolder name variable.
        :return: Fullpath string variable.
        """

        # Assemble path from given parameters with structure dir/prefix_A_B_C_D_subfolder/Summary_Episodes_X_Y_file
        subfolder_name = PathUtils.create_A_B_C_D_subfolder_name(prefix, parameters)
        x_y_name = PathUtils.create_X_Y_subfolder_name(parameters)
        csv_name = f"{x_y_name}.csv"
        return str(os.path.join(directory, subfolder_name, csv_name))

    @staticmethod
    def create_A_B_C_D_subfolder_name(prefix, parameters: dict) -> str:
        """
        Calculates and returns a subfolder name variable defined by the class attributes,
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
    def create_X_Y_subfolder_name(parameters: dict) -> str:
        """
        Calculates and returns a subfolder name variable defined by the class attributes:
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
    def return_all_result_ids(directory: str) -> list:
        """
        Returns a list of all result IDs found in the given directory.
        :param directory: Path to the directory as a string.
        :return: List of all result IDs in directory.
        """

        result_ids = []
        for file in os.listdir(directory):
            try:
                # Try to get Patient ID from filename
                result_id = int(file.split(".")[0])
                # Add filename to list
                result_ids.append(result_id)
            except TypeError as ex:
                warnings.warn(f"File: {file} has not the right format. It should be <integer>.<file extension>\n"
                              f"Error message: {ex}")
            except Exception as ex:
                warnings.warn(f"Something went wrong while file: {file} was parsed. Error {ex}")

        return result_ids