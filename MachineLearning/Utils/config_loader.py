from pathlib import Path
import yaml


def load_config(filename: str) -> dict:
    """
    Loads a YAML config from the `config/` directory.

    :param filename: Name of config file e.g. "path_config.yaml"

    :return: config as dictionary

    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert
        yaml.YAMLError: Wenn die Datei kein valides YAML ist
    """
    # Searches config directory in relation to this directory
    config_dir = Path(__file__).parent.parent / "Configs"
    config_path = config_dir / filename

    # Check if valid config path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # open config as dict
    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML error while parsing {filename}: {e}")

    return config
