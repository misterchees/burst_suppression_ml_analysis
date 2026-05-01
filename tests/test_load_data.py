import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from MachineLearning.IO.load_data import load_psd_with_start_end_resultid, LoadData

@pytest.fixture
def mock_path_manager():
    pm = MagicMock()
    return pm

@pytest.fixture
def load_data_instance(mock_path_manager):
    with patch('MachineLearning.IO.load_data.load_config') as mock_load_config:
        mock_load_config.return_value = {
            "psd_files": {
                "psd_freq_col": "Frequency_Hz",
                "psd_power_col": "PSD_V2_per_Hz"
            }
        }
        return LoadData(mock_path_manager)

def test_load_psd_with_start_end_resultid_correct_name(tmp_path):
    # Setup
    df_content = pd.DataFrame({"Frequency_Hz": [1, 2], "PSD_V2_per_Hz": [0.1, 0.2]})
    filename = "PSD_10_20_123.csv"
    file_path = tmp_path / filename
    df_content.to_csv(file_path, index=False)

    # Execute
    df, start, end, result_id = load_psd_with_start_end_resultid(tmp_path, filename)

    # Verify
    assert start == 10
    assert end == 20
    assert result_id == 123
    pd.testing.assert_frame_equal(df, df_content)

def test_load_psd_with_start_end_resultid_invalid_name():
    with pytest.raises(ValueError, match="no typical structure"):
        load_psd_with_start_end_resultid(Path("/tmp"), "invalid_name.csv")

def test_load_faw_times_as_df(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"A": 1, "B": 2, "C": 3, "D": 4, "X": 5, "Y": 6}
    mock_path_manager.get_path.return_value = tmp_path
    
    # Mock PathUtils as well to control the path generation
    with patch('MachineLearning.IO.load_data.PathUtils') as mock_path_utils:
        mock_path_utils.return_A_B_C_D_path.return_value = Path("A1_B2_C3_D4")
        mock_path_utils.return_X_Y_name.return_value = "X5_Y6"
        
        csv_dir = tmp_path / "A1_B2_C3_D4"
        csv_dir.mkdir(parents=True)
        csv_file = csv_dir / "X5_Y6.csv"
        df_content = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df_content.to_csv(csv_file, index=False)
        
        # Execute
        df = load_data_instance.load_faw_times_as_df(parameters)
        
        # Verify
        pd.testing.assert_frame_equal(df, df_content) # Assert that the DataFrame is equal
        mock_path_manager.get_path.assert_called_with("faw") # Assert that faw was used as the config argument


def test_load_faw_times_as_df_missing_file(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"A": 1, "B": 2, "C": 3, "D": 4, "X": 5, "Y": 6}
    mock_path_manager.get_path.return_value = tmp_path

    # Mock PathUtils as well to control the path generation
    with patch('MachineLearning.IO.load_data.PathUtils') as mock_path_utils:
        mock_path_utils.return_A_B_C_D_path.return_value = Path("A1_B2_C3_D4")
        mock_path_utils.return_X_Y_name.return_value = "X5_Y6"

        # Skip creating the CSV file to simulate a missing file
        # Provoke a FileNotFoundError when trying to load the CSV
        with pytest.raises(FileNotFoundError):
            load_data_instance.load_faw_times_as_df(parameters)

def test_load_awake_times_as_df(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"fixed_window_size": 10}
    mock_path_manager.get_path.return_value = tmp_path / "awake.csv"
    
    input_df = pd.DataFrame({
        "caseid": [5],
        "anestart": [30]
    })
    input_df.to_csv(tmp_path / "awake.csv", index=False)
    
    # Execute
    df = load_data_instance.load_awake_times_as_df(parameters, awake_cleaned=False, transition_time=10)
    
    # Verify
    # transition_time = 10, anestart = 30, start = 0 -> (30 - 0 - 10) // 10 = 2 epochs
    assert len(df) == 2
    assert df.iloc[0]["Start"] == 0
    assert df.iloc[0]["End"] == 10
    assert df.iloc[1]["Start"] == 10
    assert df.iloc[1]["End"] == 20
    assert df.iloc[0]["ResultID"] == 5


def test_load_cleaned_awake_times_as_df(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"fixed_window_size": 10}
    mock_path_manager.get_path.return_value = tmp_path / "awake.csv"

    input_df = pd.DataFrame({
        "case_id": [5],
        "start_time": [30],
        "end_time": [50]
    })
    input_df.to_csv(tmp_path / "awake.csv", index=False)

    # Execute
    df = load_data_instance.load_awake_times_as_df(parameters, awake_cleaned=True, transition_time=10)

    # Verify
    # transition_time = 0(ignored), start_time = 30, end_time = 50,
    # -> (50 - 30 - 0) // 10 = 2 epochs
    assert len(df) == 2
    assert df.iloc[0]["Start"] == 30
    assert df.iloc[0]["End"] == 40
    assert df.iloc[1]["Start"] == 40
    assert df.iloc[1]["End"] == 50
    assert df.iloc[0]["ResultID"] == 5

def test_load_json(tmp_path):
    # Setup
    data = {"key": "value"}
    json_path = tmp_path / "test.json"
    import json
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    # Execute
    loaded_data = LoadData.load_json(json_path)
    
    # Verify
    assert loaded_data == data

def test_group_epochs_by_result_id():
    # Setup
    df = pd.DataFrame({
        "Start": [0, 10, 5],
        "End": [10, 20, 15],
        "ResultID": [1, 1, 2]
    })
    
    # Execute
    result = LoadData._group_epochs_by_result_id(df)
    
    # Verify
    assert result == {
        1: [(0, 10), (10, 20)],
        2: [(5, 15)]
    }
