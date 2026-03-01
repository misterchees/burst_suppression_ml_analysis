from pathlib import Path
import yaml

# __file__ is the path to the current python script
# .resolve() makes it an absolute path
# .parent goes up one directory level

CURRENT_FILE = Path(__file__).resolve()
DEFAULT_CONFIG_DIR = CURRENT_FILE.parent.parent / "Configs"

def load_config(config_file: str) -> dict:
    """
    Loads a YAML config from the `config/` directory.

    :param config_file: Name of config file e.g. "path_config.yaml"
    :return: config as dictionary
    """

    config_path = DEFAULT_CONFIG_DIR / config_file
    # Check if config exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Open config as dict
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f) or {} # Fallback to empty dict if file is empty
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML error while parsing {config_path}: {e}")

    return config


def update_config(config_file: str, updates: dict) -> dict:
    """
    Updates existing keys in a YAML config file, forbidding new or unknown keys.

    :param config_file: Name of the config file
    :param updates: Dictionary of updated values (must match structure of existing config)
    :return: Updated config as dictionary
    """
    from collections.abc import Mapping

    config = load_config(config_file)

    def validate_and_update(base: dict, updates_: dict, path="") -> dict:
        for key, val in updates_.items():
            if key not in base:
                raise KeyError(f"Invalid update key '{path + key}': does not exist in config.")
            if isinstance(base[key], Mapping):
                if not isinstance(val, Mapping):
                    raise TypeError(f"Type mismatch at '{path + key}': expected dict, got {type(val).__name__}")
                base[key] = validate_and_update(base[key], val, path=path + key + ".")
            else:
                base[key] = val
        return base

    updated_config = validate_and_update(config, updates)

    config_path = DEFAULT_CONFIG_DIR / config_file
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(updated_config, f, sort_keys=False, allow_unicode=True)

    return load_config(config_file)


def replace_bands_in_config(filename: str, updates: dict):
    """
    Updates a YAML config file by merging updates recursively,
    but replaces 'frequency_bands' dictionary completely.
    """
    config = load_config(filename)

    def recursive_update(d, u, parent_key=None):
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict) and not (
                    parent_key == "relative_bandpower" and k == "frequency_bands"):
                # normal recursive merge
                recursive_update(d[k], v, k)
            else:
                # overwrite completely if key is 'frequency_bands'
                d[k] = v

    recursive_update(config, updates)

    config_path = DEFAULT_CONFIG_DIR / filename
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)