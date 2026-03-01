import os
import json
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

from MachineLearning.Utils.config_handler import load_config
from MachineLearning.Utils.path_utils import PathUtils

class PathManager:
    """
    Central management for project paths.
    Implements a hierarchy for resolving the base directory:
    1. Runtime argument
    2. Environment variable (EEG_BASE_DIR)
    3. .env file in project root
    4. User config file (~/.config/eeg_project/config.json)
    5. Fallback to local './data' directory
    """

    ENV_VAR_NAME = "EEG_BASE_DIR"
    USER_CONFIG_PATH = Path.home() / ".config" / "eeg_project" / "config.json"

    def __init__(self, config_yaml_file: Optional[str] = "path_config.yaml", base_dir: Optional[str] = None):
        # Load configs

        self.data_names = load_config("data_names_config.yaml")

        # 1. Load .env file (if present) to environment variables
        load_dotenv()

        # 2. Resolve Base Directory
        self.base_dir = self._resolve_base_dir(base_dir)

        # 3. Load Structure from path_config.yaml
        self.path_config = load_config(config_yaml_file)

        # 4. Ensure the base dir exists
        if not self.base_dir.exists():
            raise f"Warning: Base directory {self.base_dir} does not exist yet."

    def get_path(self, *keys: str) -> Path:
        """
        Recursively traverses the YAML structure to build the absolute path that will be outputted.
        Expects uniform YAML structure: {name: "...", children: {...}}
        """
        current_node = self.path_config["root"]
        path_components = []

        for key in keys:
            # Check if we need to go deeper into 'children' or if we are at root
            # Logic: If we are not at start, we look into 'children'
            if path_components:  # We are already inside the tree
                if "children" not in current_node:
                    raise KeyError(f"Node '{current_node.get('name')}' has no children, but key '{key}' was requested.")
                current_node = current_node["children"][key]
            else:
                # First key: check if it matches a top-level key under 'root'
                if key in current_node:
                    current_node = current_node[key]
                else:
                    raise KeyError(f"Root key '{key}' not found in configuration.")

            # Append the physical folder name
            path_components.append(current_node["name"])

        return self.base_dir.joinpath(*path_components)

    def set_persistent_base_dir(self, new_path: str):
        """
        Saves the base_dir to a user config file.
        This allows the CLI to 'remember' the location.
        """
        new_path_obj = Path(new_path).resolve()
        self.USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        config_data = {}
        if self.USER_CONFIG_PATH.exists():
            with open(self.USER_CONFIG_PATH, 'r') as f:
                try:
                    config_data = json.load(f)
                except:
                    pass

        config_data["base_dir"] = str(new_path_obj)

        with open(self.USER_CONFIG_PATH, 'w') as f:
            json.dump(config_data, f, indent=4)

        print(f"Base directory persistently set to: {new_path_obj}")

    def resolve_episode_path(self, parameters: dict, file_type: str, folder_keys: List[str],
                             is_file: bool = True, create_dirs: bool = False) -> Path:
        """
        Dispatcher method that returns the correct filepath depending on the file_type.
        Acts as a Facade to hide the underlying path complexity from calling classes.

        :param parameters: Defines the parameters used for folder/file naming.
        :param file_type: Defines the routing. Valid options: 'normal_an', 'faw', 'awake'.
        :param folder_keys: Keys that define the first part of the path from the base directory.
        :param is_file: If True, the last node of the path is a file (e.g., .csv).
        :param create_dirs: If True, it will create all necessary subdirectories.
        :return: The assembled filepath as a Path object.
        """
        if file_type == "faw":
            return self.get_complex_ml_path(
                parameters=parameters,
                folder_parts=folder_keys,
                is_file=is_file,
                create_dirs=create_dirs
            )
        elif file_type in ["awake", "normal_an"]:
            return self.get_simple_episode_path(
                parameters=parameters,
                file_type=file_type,
                folder_keys=folder_keys,
                is_file=is_file,
                create_dirs=create_dirs
            )
        else:
            raise ValueError(f"file_type '{file_type}' not recognized. Valid options: 'faw', 'awake', 'normal_an'")

    def get_simple_episode_path(self, parameters: dict, file_type: str,
                                folder_keys: List[str], is_file: bool = True,
                                create_dirs: bool = False) -> Path:
        """
        Creates a filepath for simple episodes (i.e. 'awake' or 'normal_anesthesia')
        where complex parameter nesting is not required.

        :param parameters: Dictionary containing parameters for the file name.
        :param file_type: Type of the file (e.g., 'awake' or 'normal_an').
        :param folder_keys: Keys to traverse the path_config for the target folder.
        :param is_file: If True, appends .csv to the target name.
        :param create_dirs: If True, physically creates the parent directories on disk.
        :return: The assembled filepath as a Path object.
        """
        # Get the base folder using the robust get_path method
        folder_dir = self.get_path(*folder_keys)

        # Get the file/node name
        node_name = self._return_node_name(parameters, file_type)
        if is_file:
            node_name = f"{node_name}.csv"

        final_path = folder_dir / node_name
        if create_dirs:
            target_dir = final_path.parent if is_file else final_path
            target_dir.mkdir(parents=True, exist_ok=True)

        return final_path

    def get_complex_ml_path(self, parameters: dict, folder_parts: List[str],
                            is_file: bool = True, create_dirs: bool = False,
                            run_name: Optional[str] = None) -> Path:
        """
        Creates a deeply nested filepath for FAW episodes and ML pipeline stages.
        Structure: .../folderN/<prefix>_A_B_C_D/<episode_name>_X_Y[.csv]

        Automatically injects run names for specific pipeline stages.

        :param parameters: Dictionary defining A, B, C, D, X, and Y parameters.
        :param folder_parts: Keys to traverse the path_config.
        :param is_file: If True, the final node is a .csv file.
        :param create_dirs: If True, physically creates the parent directories on disk.
        :param run_name: Optional specific run name to append.
        :return: The assembled filepath.
        """
        # Keys of stages that are stored in individual runs
        individual_run_keys = ["splits", "models", "results", "metadata_analysis"]

        # 1. Base Directory
        base_dir = self.get_path(*folder_parts)
        prefix_name = base_dir.stem

        # 2. Parameter-based Subdirectories
        dir_abcd_part = PathUtils.return_A_B_C_D_path(prefix_name, parameters)
        xy_part = PathUtils.return_X_Y_name(parameters)

        # 3. Assemble Core Path
        if is_file:
            core_path = base_dir / dir_abcd_part / f"{xy_part}.csv"
        else:
            core_path = base_dir / dir_abcd_part / xy_part

        # 4. Handle Run Name Injection
        final_path = core_path
        if run_name is not None:
            final_path = core_path / run_name
        # Retrieve run_name from parameters_config.yaml if not given explicitly
        elif any(key in folder_parts for key in individual_run_keys):
            current_run = load_config("parameters_config.yaml").get("run_name")
            if not current_run:
                raise ValueError("Expected 'run_name' in parameters_config but got None.")
            final_path = core_path / current_run

        # 5. Directory Creation Logic
        if create_dirs:
            # If the target is a file, create its parent directories. Otherwise, create the directory itself.
            target_dir = final_path.parent if is_file else final_path
            target_dir.mkdir(parents=True, exist_ok=True)

        return final_path

    def get_node_children(self, keys: List[str], return_type: str = "dict") -> List[str] | dict:
        """
        Navigates to a specific node in the YAML path config and returns information about its children.
        Based on the specified return_type, it returns a list of keys, values or a dictionary of children.

        :param keys: Path to the target node (e.g., "features").
        :param return_type: Type of return value. Options: "dict", "keys" or values.
        :return: A list of keys for the child nodes.
        """
        current_node = self.path_config["root"]

        # Traverse the tree to the requested node
        for key in keys:
            if "children" in current_node and key in current_node["children"]:
                current_node = current_node["children"][key]
            elif key in current_node:
                current_node = current_node[key]
            else:
                raise KeyError(f"Key '{key}' not found in configuration tree.")

        # Return the 'children' as dict or list of keys/values
        if "children" in current_node:
            children_node = current_node["children"]
            if return_type == "dict":
                return children_node
            elif return_type == "keys":
                return list(children_node.keys())
            elif return_type == "values":
                return list(children_node.values())
            else:
                raise ValueError(f"Invalid return_type '{return_type}'. Valid options are 'dict', 'keys', 'values'.")

        raise LookupError(f"Node at path '{keys}' has no children.")

    def _resolve_base_dir(self, runtime_arg: Optional[str]) -> Path:
        """Determines the base directory based on priority hierarchy."""

        # Priority 1: Runtime argument
        if runtime_arg:
            return Path(runtime_arg).resolve()
        print("No runtime argument provided for base directory. Trying .env file.")

        # Priority 2: Environment Variable (loaded from system or .env file)
        env_path = os.getenv(self.ENV_VAR_NAME)
        if env_path:
            return Path(env_path).resolve()
        print(f"Environment variable {self.ENV_VAR_NAME} not set. Trying user config file.")

        # Priority 3: User Config File
        if self.USER_CONFIG_PATH.exists():
            try:
                with open(self.USER_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                    if "base_dir" in config:
                        return Path(config["base_dir"]).resolve()
            except json.JSONDecodeError:
                pass  # Ignore corrupt config
        print(f"User config file {self.USER_CONFIG_PATH} not found. Trying fallback data directory.")

        # Priority 4: Fallback (inside the project folder)
        # Assuming this script is running from src/ or similar, go up to project root
        fallback_path = Path(__file__).resolve().parent.parent / "data"
        if fallback_path.exists():
            return fallback_path
        else:
            raise (f"No base directory found. Provide path, set environment variable, "
                   f"or create fallback Path: {fallback_path}")

    @staticmethod
    def _return_node_name(parameters: dict, node_type: str) -> str:
        """
        Return a formatted node name based on the fixed_window_size field of given parameters and node type.

        :param parameters: A dictionary containing the configuration parameters.
        :param node_type: Specifies the type of node. Valid values are 'awake' and 'normal_an'.

        :return: A formatted string representing the node name based on the node type and
            epoch length.

        :raises ValueError: If the provided node type is not recognized.
        """
        epoch_length = parameters["fixed_window_size"]
        if node_type == "awake":
            return f"Awake_{epoch_length}"
        elif node_type == "normal_an":
            return f"Normal_ane_{epoch_length}"
        raise ValueError(f"Unknown node type {node_type}. Valid types are 'awake' and 'normal_an'")