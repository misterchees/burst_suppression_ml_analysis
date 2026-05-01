import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from MachineLearning.IO.save_result import SaveResult

@pytest.fixture
def mock_path_manager():
    pm = MagicMock()
    return pm

@pytest.fixture
def save_result_instance(mock_path_manager):
    with patch('MachineLearning.IO.save_result.load_config') as mock_load_config:
        mock_load_config.return_value = {
            "psd_files": {
                "psd_freq_col": "Frequency_Hz",
                "psd_power_col": "PSD_V2_per_Hz"
            }
        }
        return SaveResult(mock_path_manager)

def test_save_psd_in_given_directory(save_result_instance, tmp_path):
    # Setup
    frequencies: np.ndarray = np.array([1, 2, 3])
    power: np.ndarray = np.array([0.1, 0.2, 0.3])
    start: int = 10
    end: int = 20
    result_id: int = 123
    
    # Execute
    save_result_instance.save_psd_in_given_directory(frequencies, power, start, end, result_id, tmp_path)
    
    # Verify
    expected_file = tmp_path / "PSD_10_20_123.csv"
    assert expected_file.exists()
    df = pd.read_csv(expected_file)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Frequency_Hz", "PSD_V2_per_Hz"]
    assert len(df) == 3

def test_save_psd_in_given_directory_invalid_values(save_result_instance, tmp_path):
    with pytest.raises(ValueError, match="start, end and result_id must be values"):
        # We use cast or just accept the type warning for the negative test
        save_result_instance.save_psd_in_given_directory(np.array([]), np.array([]), None, 20, 123, tmp_path) # type: ignore

def test_save_complete_eeg_psd(save_result_instance, mock_path_manager, tmp_path):
    # Setup
    mock_path_manager.get_path.return_value = tmp_path
    frequencies: np.ndarray = np.array([1, 2])
    power: np.ndarray = np.array([0.1, 0.2])
    result_id: int = 123
    
    # Pre-create the parent directories because the code might not use parents=True
    (tmp_path / "whole_EEG_PSD").mkdir()
    
    # Execute
    save_result_instance.save_complete_eeg_psd(frequencies, power, filtered=True, result_id=result_id)
    
    # Verify
    expected_file = tmp_path / "whole_EEG_PSD" / "filtered" / f"PSD_filtered_whole_EEG_{result_id}.csv"
    assert expected_file.exists()
    df = pd.read_csv(expected_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    mock_path_manager.get_path.assert_called_with("features", "psds")

def test_save_feature_summary_episode(save_result_instance, mock_path_manager, tmp_path):
    # Setup
    results = [{"val": 1}, {"val": 2}]
    feature_key = "test_feat"
    parameters = {"param": "val"}
    episode_type = "normal_an"
    
    fullpath = tmp_path / "summary.csv"
    mock_path_manager.resolve_episode_path.return_value = fullpath
    
    # Execute
    save_result_instance.save_feature_summary_episode(results, feature_key, parameters, episode_type)
    
    # Verify
    assert fullpath.exists()
    df = pd.read_csv(fullpath)
    assert len(df) == 2
    mock_path_manager.resolve_episode_path.assert_called_with(
        parameters, episode_type, ["features", feature_key], True, True
    )

def test_save_data_as_json(tmp_path):
    # Setup
    data = {"a": 1, "b": 2}
    fullpath = tmp_path / "test.json"
    
    # Execute
    SaveResult.save_data_as_json(data, fullpath)
    
    # Verify
    import json
    assert fullpath.exists()
    with open(fullpath, 'r') as f:
        loaded = json.load(f)
    assert loaded == data
