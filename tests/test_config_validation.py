import pytest
from pathlib import Path
import yaml

# Calculates the path to the real config file dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATH_CONFIG_PATH = PROJECT_ROOT / "MachineLearning" / "Configs" / "path_config.yaml"


def _validate_node(node_name: str, node_data: dict):
    """
    Recursively validates that a YAML node follows the defined schema:
    - Must be a dictionary.
    - Must contain a 'name' key (string).
    - If 'children' exists, it must be a dictionary of valid nodes.
    """
    assert isinstance(node_data, dict), f"Node '{node_name}' must be a dictionary."

    # 1. Every node must have a 'name' key
    assert "name" in node_data, f"Node '{node_name}' is missing the 'name' key."
    assert isinstance(node_data["name"], str), f"The 'name' value in node '{node_name}' must be a string."

    # 2. If the node has children, validate them recursively
    if "children" in node_data:
        assert isinstance(node_data["children"], dict), f"'children' in node '{node_name}' must be a dictionary."

        # Iterate through all child nodes and validate each
        for child_name, child_data in node_data["children"].items():
            _validate_node(child_name, child_data)


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

    # Validate the entire tree structure starting from 'root' keys
    root_keys = config_data["root"]
    for key in root_keys:
        _validate_node(key, root_keys[key])