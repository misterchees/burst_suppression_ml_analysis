import os


class Utils:

    # @staticmethod
    # def create_preprocessing_fullpath():
    #     """
    #     Sets the preprocessing fullpath variable to "result_..." with the subsequent dots
    #     as placeholder for the specific path depending on the attributes of EEGFeatureExtractor.
    #     """
    #     return Utils.create_fullpath("result")

    @staticmethod
    def create_anypath(*path_parts: str) -> str:
        """
        Creates any path build from given path_parts.
        :param path_parts: Any number of path parts in the right order to assemble them.
        :return: A path build from given path_parts as string.
        """
        return str(os.path.join(*path_parts))

    @staticmethod
    def create_fullpath(directory, prefix, bis_threshold: int, mac_threshold: float, min_episode_length: int,
                        refractory_time: int, merged_episodes: bool, overlap: float, fixed_window_size: int) -> str:
        """
        Calculates and returns a fullpath string variable defined by the class attributes merged_episodes,
        refractory time, min_episode_length, mac_threshold, bis_threshold, overlap and fixed_window_size.
        Example: result_70_080_20_5\\Summary_Episodes_20_000.csv

        :param directory: Initial Directory for the fullpath.
        :param prefix: Prefix of the subfolder name variable.
        :param bis_threshold: int for the bis threshold.
        :param mac_threshold: float for the mac threshold.
        :param min_episode_length: int for the min episode length.
        :param refractory_time: int for the refractory time between episodes.
        :param merged_episodes: bool flag for the prefix of the file/directory of X_Y_names.
        :param overlap: float for the overlap threshold.
        :param fixed_window_size: int for the fixed window size.
        :return: Fullpath string variable.
        """

        # Assemble path from given parameters with structure dir/prefix_A_B_C_D_subfolder/Summary_Episodes_X_Y_file
        subfolder_name = Utils.create_A_B_C_D_subfolder_name(prefix, bis_threshold,
                                                             mac_threshold, min_episode_length, refractory_time)
        x_y_name = Utils.create_X_Y_subfolder_name(merged_episodes, overlap, fixed_window_size)
        csv_name = f"{x_y_name}.csv"
        return str(os.path.join(directory, subfolder_name, csv_name))

    @staticmethod
    def create_A_B_C_D_subfolder_name(prefix, bis_threshold: int, mac_threshold: float,
                                      min_episode_length: int, refractory_time: int) -> str:
        """
        Calculates and returns a subfolder name variable defined by the class attributes,
        refractory time, min_episode_length, mac_threshold and bis_threshold.
        Example: result_70_080_20_5

        :param prefix: Prefix of the subfolder name variable.
        :param bis_threshold: int for the bis threshold.
        :param mac_threshold: float for the mac threshold.
        :param min_episode_length: int for the min episode length.
        :param refractory_time: int for the refractory time between episodes.

        :return: Subfolder string variable.
        """
        # leave 2 digits after decimal point and remove it afterward: 0.5 -> 050, 0.25 -> 025 etc.
        mac_threshold = f"{mac_threshold:.2f}".replace(".", "")
        return f"{prefix}_{bis_threshold}_{mac_threshold}_{min_episode_length}_{refractory_time}"

    @staticmethod
    def create_X_Y_subfolder_name(merged_episodes: bool, overlap: float, fixed_window_size: int) -> str:
        """
        Calculates and returns a subfolder name variable defined by the class attributes:
        merged_episodes, fixed_window_size and overlap.
        Example: Summary_Episode_20_000

        :param merged_episodes: bool flag for the prefix of the file/directory of X_Y_names.
        :param overlap: float for the overlap threshold.
        :param fixed_window_size: int for the fixed window size.
        :return: Subfolder string variable.
        """

        episode_name = "Summary_Episodes"
        if merged_episodes:
            episode_name = "Summary_Merged_Episodes"
        # leave 2 digits after decimal point and remove it afterward: 0.5 -> 050, 0.25 -> 025 etc.
        overlap = f"{overlap:.2f}".replace(".", "")
        return f"{episode_name}_{fixed_window_size}_{overlap}"
