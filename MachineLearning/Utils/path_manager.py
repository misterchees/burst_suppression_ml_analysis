import os
import yaml
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Dynamically calculate the project root relative to this file
# __file__ is the path to the current python script
# .resolve() makes it an absolute path
# .parent goes up one directory level
CURRENT_FILE = Path(__file__).resolve()

ML_DIR = CURRENT_FILE.parent.parent
DEFAULT_CONFIG_PATH = ML_DIR / "Configs" / "path_config.yaml"

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

    def __init__(self, config_yaml_path: Optional[str | Path] = None, base_dir: Optional[str] = None):
        # 1. Load .env file (if present) to environment variables
        load_dotenv()

        # 2. Set default config path if none provided
        if config_yaml_path is None:
            config_yaml_path = DEFAULT_CONFIG_PATH
        else:
            config_yaml_path = Path(config_yaml_path)

        # 2. Resolve Base Directory
        self.base_dir = self._resolve_base_dir(base_dir)

        # 3. Load Structure
        self.structure = self._load_yaml(config_yaml_path)

        # 4. Ensure the base dir exists
        if not self.base_dir.exists():
            raise f"Warning: Base directory {self.base_dir} does not exist yet."

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
    def _load_yaml(yaml_path: Path) -> Dict[str, Any]:
        """
        Loads the YAML configuration safely from a given Path.

        :param yaml_path: pathlib.Path object pointing to the config file.
        :return: Dictionary containing the parsed YAML structure.
        """
        # Use Path for robust file reading
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_path(self, *keys: str) -> Path:
        """
        Recursively traverses the YAML structure to build the absolute path that will be outputted.
        Expects uniform YAML structure: {name: "...", children: {...}}
        """
        current_node = self.structure["root"]
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