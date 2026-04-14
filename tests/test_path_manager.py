import pytest
import os
import json
from pathlib import Path
from typing import Dict
from unittest.mock import patch
from MachineLearning.Utils.path_manager import PathManager


@pytest.fixture
def mock_path_config() -> Dict:
    """Provides a sample path configuration for testing."""
    return {
        "root": {
            "initial_data": {
                "name": "Initial_data",
                "children": {
                    "raw_eeg_mat": {
                        "name": "vitalDB_mat_EEG"
                    }
                }
            },
            "features": {
                "name": "Features",
                "children": {
                    "psds": {
                        "name": "PSD"
                    },
                    "mean": {
                        "name": "Mean"
                    }
                }
            },
            "test_and_train_data": {
                "name": "Test_and_train",
                "children": {
                    "feature_sets": {
                        "name": "Feature_sets"
                    },
                    "splits": {
                        "name": "Splits"
                    }
                }
            },
            "results": {
                "name": "ML_Results",
                "children": {
                    "svm": {
                        "name": "SVM"
                    }
                }
            }
        }
    }


@pytest.fixture
def mock_parameters_config() -> Dict:
    """Provides a sample parameters configuration for testing."""
    return {
        "run_name": "test_run_123"
    }


@pytest.fixture
def temp_base_dir(tmp_path: Path) -> Path:
    """Provides a temporary base directory."""
    base = tmp_path / "data"
    base.mkdir()
    return base


@pytest.fixture
def path_manager(temp_base_dir: Path, mock_path_config: Dict) -> PathManager:
    """Provides a PathManager instance with mocked config and temp base dir."""
    # Replace the load_config call with the defined mock
    with patch("MachineLearning.Utils.path_manager.load_config", return_value=mock_path_config):
        with patch("MachineLearning.Utils.path_manager.load_dotenv"):
            return PathManager(base_dir=str(temp_base_dir))


def test_resolve_base_dir_priority(tmp_path: Path):
    """Tests the priority of base directory resolution."""
    # Priority 1: Runtime argument
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    with patch("MachineLearning.Utils.path_manager.load_config"):
        with patch("MachineLearning.Utils.path_manager.load_dotenv"):
            pm = PathManager(base_dir=str(runtime_dir))
    assert pm.base_dir == runtime_dir.resolve()

    # Priority 2: Environment variable
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    with patch.dict(os.environ, {"EEG_BASE_DIR": str(env_dir)}):
        with patch("MachineLearning.Utils.path_manager.load_config"):
            with patch("MachineLearning.Utils.path_manager.load_dotenv"):
                pm = PathManager()
    assert pm.base_dir == env_dir.resolve()

    # Priority 3: User Config File
    user_config_dir = tmp_path / "user_config"
    user_config_dir.mkdir()
    # Set up mocked config file
    user_config_file = tmp_path / ".user_config.json"
    with open(user_config_file, "w") as f:
        json.dump({"base_dir": str(user_config_dir)}, f)

    with patch.object(PathManager, "USER_CONFIG_PATH", user_config_file):
        with patch.dict(os.environ, {}, clear=True): # clear env to ensure priority is not from env
            with patch("MachineLearning.Utils.path_manager.load_config"):
                with patch("MachineLearning.Utils.path_manager.load_dotenv"):
                    pm = PathManager()
    assert pm.base_dir == user_config_dir.resolve()


def test_get_path(path_manager: PathManager, temp_base_dir: Path):
    """Tests navigation through the path tree."""
    # Test root
    root_path = path_manager.get_path()
    assert root_path == temp_base_dir

    # Test top-level key
    initial_data_path = path_manager.get_path("initial_data")
    assert initial_data_path == temp_base_dir / "Initial_data"

    # Test nested key
    raw_eeg_path = path_manager.get_path("initial_data", "raw_eeg_mat")
    assert raw_eeg_path == temp_base_dir / "Initial_data" / "vitalDB_mat_EEG"

    # Test invalid key
    with pytest.raises(KeyError):
        path_manager.get_path("non_existent")

    with pytest.raises(KeyError):
        path_manager.get_path("initial_data", "non_existent_child")


def test_get_simple_episode_path(path_manager: PathManager, temp_base_dir: Path):
    """Tests path generation for simple episodes (awake/normal_an)."""
    params = {"fixed_window_size": 20}

    # Awake
    awake_path = path_manager.get_simple_episode_path(params, "awake", ["initial_data", "raw_eeg_mat"])
    assert awake_path == temp_base_dir / "Initial_data" / "vitalDB_mat_EEG" / "Awake_20.csv"

    # Normal anesthesia
    normal_an_path = path_manager.get_simple_episode_path(params, "normal_an", ["initial_data", "raw_eeg_mat"])
    assert normal_an_path == temp_base_dir / "Initial_data" / "vitalDB_mat_EEG" / "Normal_ane_20.csv"

    # Invalid type
    with pytest.raises(ValueError):
        path_manager.get_simple_episode_path(params, "invalid", ["initial_data"])

    # Directory creation
    new_path = path_manager.get_simple_episode_path(params, "awake", ["initial_data"], create_dirs=True)
    assert new_path.parent.exists()


def test_get_complex_ml_path(path_manager: PathManager, temp_base_dir: Path, mock_parameters_config: Dict):
    """Tests path generation for complex ML stages (FAW)."""
    params = {
        "mac_threshold": 0.5,
        "bis_threshold": 70,
        "min_episode_length": 20,
        "refractory_time": 5,
        "merged_episodes": True,
        "fixed_window_size": 20,
        "overlap": 0.25
    }

    # Test FAW path with missing run_name key
    with pytest.raises(ValueError):
        with patch("MachineLearning.Utils.path_manager.load_config", return_value={}):
            path_manager.get_complex_ml_path(params, ["test_and_train_data", "splits"], False)

    # Test FAW path with missing run_name value
    with pytest.raises(ValueError):
        with patch("MachineLearning.Utils.path_manager.load_config", return_value={"run_name": ""}):
            path_manager.get_complex_ml_path(params, ["test_and_train_data", "splits"], False)

    # Test FAW path without explicit run_name (is_file=False)
    with patch("MachineLearning.Utils.path_manager.load_config", return_value=mock_parameters_config):
        faw_path = path_manager.get_complex_ml_path(params, ["test_and_train_data", "splits"], False)
        # Expected structure: base_dir / abcd_part / xy_part / run_name.csv
        expected_path = (temp_base_dir / "Test_and_train" / "Splits" / "Splits_70_050_20_5" /
                         "Summary_Merged_Episodes_20_025" / "test_run_123")
        assert faw_path == expected_path

    # Test FAW path with explicit run_name (is_file=False)
    faw_path_with_run = path_manager.get_complex_ml_path(params, ["test_and_train_data", "splits"], False,
                                                       run_name="explicit_run")
    expected_path_with_run = (temp_base_dir / "Test_and_train" / "Splits" / "Splits_70_050_20_5" /
                              "Summary_Merged_Episodes_20_025" / "explicit_run")
    assert faw_path_with_run == expected_path_with_run

    # Test FAW path (is_file=True)
    with patch("MachineLearning.Utils.path_manager.load_config", return_value=mock_parameters_config):
        faw_path = path_manager.get_complex_ml_path(params, ["features", "mean"], True)
        # Expected structure: base_dir / abcd_part / xy_part / run_name.csv
        expected_path = (temp_base_dir / "Features" / "Mean" / "Mean_70_050_20_5" /
                         "Summary_Merged_Episodes_20_025").with_suffix(".csv")
        assert faw_path == expected_path


def test_get_related_paths(path_manager: PathManager, temp_base_dir: Path):
    """Tests retrieving related files (splits/results)."""
    params = {
        "mac_threshold": 0.5, "bis_threshold": 70, "min_episode_length": 20,
        "refractory_time": 5, "merged_episodes": True, "fixed_window_size": 20, "overlap": 0.25
    }
    run_name = "test_run"

    # Set up splits folder for test_and_train_data
    splits_folder = (temp_base_dir / "Test_and_train" / "Splits" / "Splits_70_050_20_5" /
                     "Summary_Merged_Episodes_20_025" / run_name)
    splits_folder.mkdir(parents=True)
    (splits_folder / "any_prefix_train_split.csv").touch()
    (splits_folder / "any_prefix2_test_split.csv").touch()
    (splits_folder / "ignored.csv").touch()

    related_splits = path_manager.get_related_paths(params, run_name, ["test_and_train_data", "splits"])
    assert len(related_splits) == 2
    assert any(p.name == "any_prefix_train_split.csv" for p in related_splits)
    assert any(p.name == "any_prefix2_test_split.csv" for p in related_splits)

    # Set up results folder
    results_folder = (temp_base_dir / "ML_Results" / "SVM" / "SVM_70_050_20_5" /
                      "Summary_Merged_Episodes_20_025" / run_name)
    results_folder.mkdir(parents=True)

    # Test for empty results folder
    with pytest.raises(FileNotFoundError):
        path_manager.get_related_paths(params, run_name, ["results", "svm"])

    # Test for non-empty results folder
    (results_folder / "1_6_test_split_full_and_pred.csv").touch()
    (results_folder / "5_6_test_split_full_and_pred.csv").touch()
    (results_folder / "ignored.csv").touch()

    related_results = path_manager.get_related_paths(params, run_name, ["results", "svm"])
    assert len(related_results) == 2
    assert any(p.name == "1_6_test_split_full_and_pred.csv" for p in related_results)
    assert any(p.name == "5_6_test_split_full_and_pred.csv" for p in related_results)

    # Invalid folder keys for related_paths
    with pytest.raises(ValueError):
        path_manager.get_related_paths(params, run_name, ["features", "psds"])


def test_resolve_episode_path(path_manager: PathManager, temp_base_dir: Path):
    """Tests the dispatcher method resolve_episode_path."""
    params = {
        "mac_threshold": 0.5, "bis_threshold": 70, "min_episode_length": 20,
        "refractory_time": 5, "merged_episodes": True, "fixed_window_size": 20, "overlap": 0.25
    }

    # Test FAW (goes to get_complex_ml_path)
    # Note: since 'splits' is in individual_run_keys, we need to mock parameters_config for run_name
    with patch("MachineLearning.Utils.path_manager.load_config", return_value={"run_name": "test_run"}):
        path = path_manager.resolve_episode_path(params, "faw", ["test_and_train_data", "splits"], False)
        assert "Splits_70_050_20_5" in str(path)
        assert "test_run" in str(path)

    # Test Awake (goes to get_simple_episode_path)
    path = path_manager.resolve_episode_path(params, "awake", ["initial_data", "raw_eeg_mat"])
    assert "Awake_20.csv" in str(path)

    # Test Invalid type
    with pytest.raises(ValueError):
        path_manager.resolve_episode_path(params, "invalid", ["initial_data"])


def test_get_all_patient_ids(path_manager: PathManager, temp_base_dir: Path):
    """Tests extraction of patient IDs from file names."""

    # Valid file names
    target_dir = temp_base_dir / "Initial_data" / "vitalDB_mat_EEG"
    target_dir.mkdir(parents=True)
    (target_dir / "101.mat").touch()
    (target_dir / "202.mat").touch()
    (target_dir / "not_an_id.txt").touch()

    patient_ids = path_manager.get_all_patient_ids(["initial_data", "raw_eeg_mat"])
    assert sorted(patient_ids) == [101, 202]


def test_set_persistent_base_dir(path_manager: PathManager, tmp_path: Path):
    """Tests setting the persistent base directory in a config file."""
    user_config_file = tmp_path / "test_config.json"
    new_base_dir = tmp_path / "new_base"
    new_base_dir.mkdir()

    with patch.object(PathManager, "USER_CONFIG_PATH", user_config_file):
        path_manager.set_persistent_base_dir(str(new_base_dir))

    assert user_config_file.exists()
    with open(user_config_file, "r") as f:
        config = json.load(f)
        assert config["base_dir"] == str(new_base_dir.resolve())


def test_get_node_children(path_manager: PathManager):
    """Tests navigation to a node and returning its children in various formats."""
    # Test 'dict' return type (default)
    children_dict = path_manager.get_node_children(["features"])
    assert isinstance(children_dict, dict)
    assert "psds" in children_dict
    assert "mean" in children_dict

    # Test 'keys' return type
    children_keys = path_manager.get_node_children(["features"], return_type="keys")
    assert sorted(children_keys) == sorted(["psds", "mean"])

    # Test 'values' return type
    children_values = path_manager.get_node_children(["features"], return_type="values")
    assert isinstance(children_values, list)
    assert any(v["name"] == "PSD" for v in children_values)

    # Test deeply nested children
    nested_children = path_manager.get_node_children(["initial_data"], return_type="keys")
    assert "raw_eeg_mat" in nested_children

    # Error: Key not found
    with pytest.raises(KeyError):
        path_manager.get_node_children(["non_existent"])

    # Error: Node has no children
    with pytest.raises(LookupError):
        path_manager.get_node_children(["features", "psds"])

    # Error: Invalid return type
    with pytest.raises(ValueError):
        path_manager.get_node_children(["features"], return_type="invalid")


