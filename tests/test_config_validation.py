import pytest
from pathlib import Path
import yaml

# Important: Calculate the path to the real config file dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATH_CONFIG_PATH = PROJECT_ROOT / "MachineLearning" / "Configs" / "path_config.yaml"


def test_path_config_is_valid_yaml():
    """
    Validates that the actual project configuration file exists,
    is valid YAML, and contains the required top-level structure.
    """
    # Assert that the file actually exists where we expect it
    assert PATH_CONFIG_PATH.exists(), f"Configuration file not found at {PATH_CONFIG_PATH}"

    # Try parsing it to catch YAML syntax errors (e.g., indentation issues)
    with open(PATH_CONFIG_PATH, 'r', encoding='utf-8') as file:
        try:
            config_data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            pytest.fail(f"Real config file contains invalid YAML syntax: {exc}")

    # Check if the mandatory root structure exists
    assert isinstance(config_data, dict), "Config should be a dictionary"
    assert "root" in config_data, "Config must contain the top-level key 'root'"