import pytest
from pathlib import Path

from MachineLearning.Utils.path_manager import PathManager


@pytest.fixture
def setup_path_manager(tmp_path: Path):
    """
    Creates a temporary base directory and a dummy YAML configuration
    to test the PathManager safely in isolation.
    """
    # Create a mock base directory for our fake data
    mock_base_dir = tmp_path / "mock_data"
    mock_base_dir.mkdir()

    # Create a mock YAML config mimicking the project's structure
    mock_yaml_path = tmp_path / "mock_config.yaml"
    mock_yaml_content = """
    root:
      initial_data:
        name: "Initial_data"
        children:
          raw_eeg_mat:
            name: "vitalDB_mat_EEG"
      features:
        name: "Features"
    """
    mock_yaml_path.write_text(mock_yaml_content, encoding='utf-8')

    # Initialize the PathManager with the mock setup
    pm = PathManager(config_yaml_path=mock_yaml_path, base_dir=str(mock_base_dir))

    return pm, mock_base_dir

def test_get_path_valid_keys_first_level(setup_path_manager):
    """
    Test if get_path correctly resolves existing keys to a valid pathlib.Path object in first level.
    """
    # Unpack the fixture
    pm, mock_base_dir = setup_path_manager

    # Try to resolve a path
    resolved_path = pm.get_path("features")

    # Check if the constructed path matches expectations
    expected_path = mock_base_dir / "Features"

    assert resolved_path == expected_path
    assert isinstance(resolved_path, Path)


def test_get_path_valid_keys_multiple_levels(setup_path_manager):
    """
    Test if get_path correctly resolves existing keys to a valid pathlib.Path object over multiple levels.
    """
    # Unpack the fixture
    pm, mock_base_dir = setup_path_manager

    # Act: Try to resolve a path
    resolved_path = pm.get_path("initial_data", "raw_eeg_mat")

    # Assert: Check if the constructed path matches expectations
    expected_path = mock_base_dir / "Initial_data" / "vitalDB_mat_EEG"

    assert resolved_path == expected_path
    assert isinstance(resolved_path, Path)


def test_get_path_invalid_key(setup_path_manager):
    """
    Test if get_path correctly raises a KeyError when an unknown key is provided.
    """
    pm, _ = setup_path_manager

    # Expecting KeyError to be raised
    with pytest.raises(KeyError) as exc_info:
        pm.get_path("initial_data", "this_key_does_not_exist")

    # Check for key error message
    assert "this_key_does_not_exist" in str(exc_info.value)

def test_get_path_invalid_key_in_leaf(setup_path_manager):
    """
    Test if get_path correctly raises a KeyError when a key is provided for a leaf.
    """
    pm, _ = setup_path_manager

    # Expecting KeyError to be raised
    with pytest.raises(KeyError) as exc_info:
        pm.get_path("features", "this_key_does_not_exist")

    # Check for key error message
    assert "has no children, but key" in str(exc_info.value)


def test_get_path_missing_name_in_node(tmp_path: Path):
    """
    Tests if get_path handles if 'name' key is missing in a node.
    """
    # Create a faulty YAML
    faulty_yaml = tmp_path / "faulty.yaml"
    yaml_content = """
        root:
          some_dir:
            not_name: 'not a name'
        """
    faulty_yaml.write_text(yaml_content, encoding='utf-8')
    pm = PathManager(config_yaml_path=faulty_yaml, base_dir=str(tmp_path))

    # Expecting KeyError to be raised
    with pytest.raises(KeyError) as exc_info:
        pm.get_path("some_dir")

    assert "name" in str(exc_info.value)


def test_get_path_missing_root_in_yaml(tmp_path: Path):
    """
    Tests if get_path handles if 'root' is missing.
    """
    # Create a faulty YAML
    faulty_yaml = tmp_path / "faulty.yaml"
    yaml_content = """
                wrong_entry:
                  some_dir: 'some_value'
                """
    faulty_yaml.write_text("wrong_entry:\n  some_dir: 'name'", encoding='utf-8')
    pm = PathManager(config_yaml_path=faulty_yaml, base_dir=str(tmp_path))

    # Act & Assert: PathManager should fail when trying to find 'root' in __init__
    with pytest.raises(KeyError) as exc_info:
        pm.get_path("some_dir")

    assert "root" in str(exc_info.value)


def test_load_yaml_static_method(tmp_path: Path):
    """
    Test the static _load_yaml method independently to ensure it parses correctly.
    """
    # Arrange: Create a simple YAML file
    test_yaml_path = tmp_path / "test.yaml"
    yaml_content = """
            root:
              test_key: 'test_value'
            """
    test_yaml_path.write_text(yaml_content, encoding='utf-8')

    # Act: Call the static method directly from the class
    result = PathManager._load_yaml(test_yaml_path)

    # Assert
    assert isinstance(result, dict)
    assert result["root"]["test_key"] == "test_value"