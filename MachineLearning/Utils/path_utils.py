import os


class PathUtils:
    @staticmethod
    def return_anypath(*path_parts: str) -> str:
        """
        Creates any path build from given path_parts.
        :param path_parts: Any number of path parts in the right order to assemble them.
        :return: A path build from given path_parts as string.
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
        :param prefix: prefix of abcd folder
        :param parameters: parameters that decide about the variables.
        :return: <prefix>_A_B_C_D\\<Episode>_X_Y
        """
        abcd_folder = PathUtils.return_A_B_C_D_name(prefix, parameters)
        xy_folder = PathUtils.return_X_Y_name(parameters)
        return str(os.path.join(abcd_folder, xy_folder))

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