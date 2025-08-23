from pathlib import Path
import yaml


def load_config(filename: str) -> dict:
    """
    Loads a YAML config from the `config/` directory, which is relative to this directory.

    :param filename: Name of config file e.g. "path_config.yaml"
    :return: config as dictionary
    """
    # Searches config directory in relation to this directory
    config_dir = Path(__file__).parent.parent / "Configs"
    config_path = config_dir / filename

    # Check if valid config path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Open config as dict
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML error while parsing {filename}: {e}")

    return config


def update_config(filename: str, updates: dict) -> dict:
    """
    Updates existing keys in a YAML config file, forbidding new or unknown keys.

    :param filename: Name of the YAML file inside Configs/
    :param updates: Dictionary of updated values (must match structure of existing config)
    :return: Updated config as dictionary
    """
    from collections.abc import Mapping

    config_dir = Path(__file__).parent.parent / "Configs"
    config_path = config_dir / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

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

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(updated_config, f, sort_keys=False, allow_unicode=True)

    return load_config(filename)


def replace_bands_in_config(filename, updates):
    """
    Updates a YAML config file by merging updates recursively,
    but replaces 'frequency_bands' dictionary completely.
    """
    config_dir = Path(__file__).parent.parent / "Configs"
    config_path = config_dir / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

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

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)