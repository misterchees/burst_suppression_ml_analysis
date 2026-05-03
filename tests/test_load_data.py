import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

from MachineLearning.Evaluation.meta_fold_analyzer import MetaFoldAnalyzer
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
            },
            "eeg_files": {
                "eeg_channels": ["1", "2"],
                "eeg_fs": "fs_key",
                "eeg_rawEEG": "raw_key"
            }
        }
        return LoadData(mock_path_manager)

# --- Tests for load_psd_with_start_end_resultid ---
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
    # False Prefix
    with pytest.raises(ValueError, match="no typical structure"):
        load_psd_with_start_end_resultid(Path("/tmp"), "falsePrefix_1_2_3.csv")

    # False structure
    with pytest.raises(ValueError, match="no typical structure"):
        load_psd_with_start_end_resultid(Path("/tmp"), "1_2_3.csv")

# --- Tests for load_faw_times_as_df ---
def test_load_faw_times_as_df(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = MagicMock() # Mock the parameters since PathUtils return value is also mocked
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
    parameters = MagicMock() # Mock the parameters since PathUtils return value is also mocked
    mock_path_manager.get_path.return_value = tmp_path

    # Mock PathUtils as well to control the path generation
    with patch('MachineLearning.IO.load_data.PathUtils') as mock_path_utils:
        mock_path_utils.return_A_B_C_D_path.return_value = Path("A1_B2_C3_D4")
        mock_path_utils.return_X_Y_name.return_value = "X5_Y6"

        # Skip creating the CSV file to simulate a missing file
        # Provoke a FileNotFoundError when trying to load the CSV
        with pytest.raises(FileNotFoundError):
            load_data_instance.load_faw_times_as_df(parameters)

# --- Tests for load_awake_times_as_df ---
def test_load_awake_times_as_df(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"fixed_window_size": 10} # Only relevant parameter for this function
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
    parameters = {"fixed_window_size": 10} # Only relevant parameter for this function
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

# --- Tests for _load_filtered_eeg_data ---
def test_load_filtered_eeg_data(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    csv_dir = tmp_path
    mock_path_manager.get_path.return_value = csv_dir

    # Mock input data
    in_df = pd.DataFrame({
        "1": [1.0, 2.0, 3.0],
        "2": [4.0, 5.0, 6.0]
    })
    in_fs = 128

    fullpath = csv_dir / "1.csv"
    # Write the input data to a CSV file in the correct format
    # Header line with fs, then the data
    with open(fullpath, 'w', newline='') as f:
        f.write(f"# fs = {in_fs}\n")
        in_df.to_csv(f, index=False)

    # Execute
    out_fs, out_np_arr = load_data_instance._load_filtered_eeg_data(1)

    # Verify
    assert out_fs == in_fs # Check if fs is read correctly
    assert out_np_arr.shape == (3, 2) # Assert Array Shape
    np.testing.assert_array_almost_equal(in_df[["1", "2"]].to_numpy(), out_np_arr) # Assert Array Equal

def test_load_filtered_eeg_data_errors(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    csv_dir = tmp_path
    mock_path_manager.get_path.return_value = csv_dir

    # No file created -> FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_data_instance._load_filtered_eeg_data(1)

    # Malformed fs header line
    fullpath_malformed = csv_dir / "1.csv"
    with open(fullpath_malformed, 'w', newline='') as f:
        f.write(f"malformed header\n")
        pd.DataFrame({"1": [1], "2": [2]}).to_csv(f, index=False)

    with pytest.raises(ValueError, match="First line does not contain sampling rate"):
        load_data_instance._load_filtered_eeg_data(1)

    # Missing fs in header
    fullpath_no_fs = csv_dir / "2.csv"
    pd.DataFrame({"1": [1], "2": [2]}).to_csv(fullpath_no_fs, index=False)

    with pytest.raises(ValueError, match="First line does not contain sampling rate"):
        load_data_instance._load_filtered_eeg_data(2)

# --- Tests for load_eeg_epochs_from_csv ---
def test_load_eeg_epochs_from_csv(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    folder_keys = MagicMock() # Not relevant, since the path is mocked
    mock_path_manager.get_path.return_value = tmp_path
    result_id = 123
    fs = 100
    epochs = [(1, 2), (3, 4)] # seconds
    channel = 1
    
    # Create mock EEG CSV
    # 1s to 2s -> rows 100 to 200 (approx) + offset
    # 3s to 4s -> rows 300 to 400 (approx) + offset
    # Mocked CSV contains enough data for 5 seconds
    data_len = 500
    df = pd.DataFrame({
        "1": np.arange(data_len, dtype=float),
        "2": np.arange(data_len, dtype=float) * 2
    })
    # Write the CSV accordingly with the sampling rate in the header
    filepath = tmp_path / f"{result_id}.csv"
    with open(filepath, 'w', newline='') as f:
        f.write(f"# fs = {fs}\n")
        df.to_csv(f, index=False)
        
    # Execute
    out_fs, segments = load_data_instance.load_eeg_epochs_from_csv(result_id, epochs, channel, folder_keys)
    
    # Verify
    assert out_fs == fs
    assert len(segments) == 2
    assert (1, 2) in segments
    assert (3, 4) in segments
    
    # Expected segment 1: 1s * 100 = sample 100 to 2s * 100 = sample 200
    expected_seg1 = df.iloc[100:200]["1"].to_numpy()
    np.testing.assert_array_equal(segments[(1, 2)], expected_seg1)
    
    expected_seg2 = df.iloc[300:400]["1"].to_numpy()
    np.testing.assert_array_equal(segments[(3, 4)], expected_seg2)

def test_load_eeg_epochs_from_csv_errors(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    folder_keys = MagicMock()
    mock_path_manager.get_path.return_value = tmp_path
    result_id = 999
    
    # 1. FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_data_instance.load_eeg_epochs_from_csv(result_id, [(0, 1)], 1, folder_keys)
        
    # 2. Malformed Header
    filepath = tmp_path / f"{result_id}.csv"
    with open(filepath, 'w', newline='') as f:
        f.write("wrong header\n")
        pd.DataFrame({"1": [1]}).to_csv(f, index=False)
        
    with pytest.raises(ValueError, match="Missing or malformed sampling rate"):
        load_data_instance.load_eeg_epochs_from_csv(result_id, [(0, 1)], 1, folder_keys)

# --- Tests for load_global_outliers ---
def test_load_global_outliers(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = MagicMock()
    outlier_df = pd.DataFrame({"group": [1, 2, 3]})
    
    mock_path_manager.get_complex_ml_path.return_value = tmp_path
    
    # Mock for epoch type
    epoch_file = tmp_path / "global_epoch_outliers.csv"
    outlier_df.to_csv(epoch_file, index=False)
    
    # Mock for patient type
    patient_df = pd.DataFrame({"group": [10, 20]})
    patient_file = tmp_path / "global_patient_outliers.csv"
    patient_df.to_csv(patient_file, index=False)
    
    # Execute & Verify Epoch
    res_epoch = load_data_instance.load_global_outliers(parameters, "epoch")
    pd.testing.assert_frame_equal(res_epoch, outlier_df)
    
    # Execute & Verify Patient
    res_patient = load_data_instance.load_global_outliers(parameters, "patient_id")
    pd.testing.assert_frame_equal(res_patient, patient_df)
    
    # Test Invalid Type
    with pytest.raises(ValueError, match="Invalid outlier type"):
        load_data_instance.load_global_outliers(parameters, "invalid")

# --- Tests for load_further_results ---
def test_load_further_results(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    hyperparameters = MagicMock()
    analysis_key = MagicMock()
    
    mock_path_manager.get_complex_ml_path.return_value = tmp_path
    
    # 1. Test CSV loading
    csv_df = pd.DataFrame({"a": [1], "b": [2]})
    csv_filename = "result.csv"
    csv_df.to_csv(tmp_path / csv_filename, index=False)
    
    res_csv = load_data_instance.load_further_results(hyperparameters, analysis_key, csv_filename)
    pd.testing.assert_frame_equal(res_csv, csv_df)
    
    # 2. Test JSON loading
    json_data = {"x": 10, "y": [1, 2, 3]}
    json_filename = "result.json"
    with patch.object(LoadData, 'load_json', return_value=json_data):
        res_json = load_data_instance.load_further_results(hyperparameters, analysis_key, json_filename)
        assert res_json == json_data
        
    # 3. Test invalid extension
    with pytest.raises(ValueError, match="Invalid result_file extension"):
        load_data_instance.load_further_results(hyperparameters, analysis_key, "result.txt")

# --- Tests for load_run_data ---
def test_load_run_data(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    hyperparameters = MagicMock()
    run_name = "test_run"
    model_key = MagicMock()
    
    mock_path_manager.get_complex_ml_path.return_value = tmp_path
    
    metadata = {"run_id": "test_run", "accuracy": 0.95}
    json_file = tmp_path / f"{run_name}.json"
    other_file = tmp_path / f"{run_name}.txt"
    other_file2 = tmp_path / "other_file.json"
    
    with patch('MachineLearning.IO.load_data.PathUtils.list_files_in_folder') as mock_list:
        mock_list.return_value = ([other_file, other_file2, json_file], 3)
        
        with patch.object(LoadData, 'load_json', return_value=metadata):
            # Execute
            result = load_data_instance.load_run_data(hyperparameters, run_name, model_key)
            
            # Verify
            assert result == metadata
            mock_list.assert_called_once()

def test_load_run_data_not_found(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    hyperparameters = MagicMock()
    run_name = "missing_run"
    model_key = MagicMock()

    mock_path_manager.get_complex_ml_path.return_value = tmp_path
    
    with patch('MachineLearning.IO.load_data.PathUtils.list_files_in_folder') as mock_list:
        mock_list.return_value = ([], 0)
        
        # Execute & Verify
        with pytest.raises(FileNotFoundError, match="No matching file for run_name"):
            load_data_instance.load_run_data(hyperparameters, run_name, model_key)

# --- Tests for load_json ---
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

# --- Tests for _group_epochs_by_result_id ---
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

# --- Tests for sample_anesthesia_epochs ---
def test_sample_anesthesia_epochs(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"fixed_window_size": 10}
    num_epochs = 5
    
    # Paths
    awake_times_path = tmp_path / "awake_times.csv"
    filtered_data_dir = tmp_path / "filtered_data"
    filtered_data_dir.mkdir()

    # Mock for both path-accessing instances in function
    mock_path_manager.get_path.side_effect = lambda *args: awake_times_path if args[0] == "awake_times" else filtered_data_dir
    
    # Mock awake_times.csv
    # Patient 1: anestart at 60s
    # Patient 2: anestart not present (will use default)
    pd.DataFrame({"caseid": [1], "anestart": [60]}).to_csv(awake_times_path, index=False)
    
    # Mock filtered EEG files (CSV)
    # File 1.csv: 128 Hz * 1000s = 128000 lines + 1 header = 128001 lines
    # Duration = 1000s. anestart=60. transition=10 -> start_limit=70.
    # safety_margin=10min=600s. end_limit = 1000 - 600 - 10 = 390.
    # Valid range [70, 390]
    eeg1_content = "header\n" + "0,0\n" * (128 * 1000)
    with open(filtered_data_dir / "1.csv", "w") as f:
        f.write(eeg1_content)
        
    # File 2.csv: 128 Hz * 1200s = 153600 lines + 1 header = 153601 lines
    # anestart default = 10min = 600s. transition=10 -> start_limit=610.
    # safety_margin=10min=600s. end_limit = 1200 - 600 - 10 = 590.
    # end_limit (590) <= start_limit (610) -> Should be skipped!
    eeg2_content = "header\n" + "0,0\n" * (128 * 1200)
    with open(filtered_data_dir / "2.csv", "w") as f:
        f.write(eeg2_content)

    # 1. Num epochs is constraint
    # Execute
    df = load_data_instance.sample_anesthesia_epochs(parameters, num_epochs)
    
    # Verify
    assert isinstance(df, pd.DataFrame)
    assert len(df) == num_epochs
    assert all(df["ResultID"] == "1") # Only 1.csv should have been sampled
    for _, row in df.iterrows():
        assert 70 <= row["Start"] <= 390 - 10 # start_points in range(start_limit, end_limit - epoch_length + 1)
        assert row["End"] == row["Start"] + 10

    # 2. Max number of epochs per eeg is the constraint, since in this test case only one eeg is effectively sampled
    # Execute
    max_epochs_per_eeg = 2
    df2 = load_data_instance.sample_anesthesia_epochs(parameters, num_epochs, epochs_per_eeg=max_epochs_per_eeg)

    # Verify
    assert len(df2) == max_epochs_per_eeg

def test_sample_anesthesia_epochs_too_few_data(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"fixed_window_size": 10}
    num_epochs = 100 # Request many epochs
    
    awake_times_path = tmp_path / "awake_times.csv"
    filtered_data_dir = tmp_path / "filtered_data"
    filtered_data_dir.mkdir()
    
    mock_path_manager.get_path.side_effect = lambda *args: awake_times_path if args[0] == "awake_times" else filtered_data_dir
    pd.DataFrame({"caseid": [1], "anestart": [0]}).to_csv(awake_times_path, index=False)
    
    # Small file: 128 * 700 lines -> 700s
    # start_limit = 0 + 10 = 10
    # end_limit = 700 - 600 - 10 = 90
    # max_possible_epochs = (90 - 10) // 10 = 8
    eeg_content = "header\n" + "0,0\n" * (128 * 700)
    with open(filtered_data_dir / "1.csv", "w") as f:
        f.write(eeg_content)
        
    # Execute
    df = load_data_instance.sample_anesthesia_epochs(parameters, num_epochs, epochs_per_eeg=50)
    
    # Verify
    assert len(df) == 8 # Only 8 epochs possible

# --- Tests for _load_eeg_vitaldb_csv ---
def test_load_eeg_vitaldb_csv(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    result_id = 123
    mock_path_manager.get_path.return_value = tmp_path
    
    # Create CSV with some NaNs
    csv_file = tmp_path / f"{result_id}.csv"
    df_content = pd.DataFrame({
        'BIS/EEG1_WAV': [1.0, np.nan, 3.0, 4.0],
        'BIS/EEG2_WAV': [np.nan, 2.0, 3.0, np.nan]
    })
    df_content.to_csv(csv_file, index=False)
    
    # Execute
    fs, data = load_data_instance._load_eeg_vitaldb_csv(result_id)
    
    # Verify
    assert fs == 128
    assert data.shape == (4, 2)
    # Check interpolation
    # EEG1: [1.0, nan, 3.0, 4.0] -> [1.0, 2.0, 3.0, 4.0]
    # EEG2: [nan, 2.0, 3.0, nan] -> [2.0, 2.0, 3.0, 3.0] (limit_direction='both')
    expected = np.array([
        [1.0, 2.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 3.0]
    ])
    np.testing.assert_array_equal(data, expected)

def test_load_eeg_vitaldb_csv_missing(load_data_instance, mock_path_manager, tmp_path):
    mock_path_manager.get_path.return_value = tmp_path
    with pytest.raises(FileNotFoundError):
        load_data_instance._load_eeg_vitaldb_csv(999)

# --- Tests for _load_raw_eeg_data ---
def test_load_raw_eeg_data_mat(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    result_id = 456
    mock_path_manager.get_path.return_value = tmp_path
    mat_file = tmp_path / f"{result_id}.mat"
    mat_file.touch()
    
    # Mock scipy.io.loadmat and data_names
    mock_mat_data = {
        'fs_key': np.array([[256]]),
        'raw_key': np.array([[1, 2], [3, 4]])
    }
    
    with patch('scipy.io.loadmat') as mock_loadmat:
        mock_loadmat.return_value = mock_mat_data
        
        # Execute
        fs, data = load_data_instance._load_raw_eeg_data(result_id)
        
        # Verify
        assert fs == 256
        np.testing.assert_array_equal(data, mock_mat_data['raw_key'])

def test_load_raw_eeg_data_fallback_to_csv(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    result_id = 789
    mock_path_manager.get_path.return_value = tmp_path
    # No .mat file exists

    in_fs = 128
    # Mock _load_eeg_vitaldb_csv
    with patch.object(LoadData, '_load_eeg_vitaldb_csv') as mock_load_csv:
        mock_load_csv.return_value = (in_fs, np.array([[0]]))
        
        # Execute
        fs, data = load_data_instance._load_raw_eeg_data(result_id)
        
        # Verify
        assert fs == in_fs
        mock_load_csv.assert_called_once_with(result_id) # Assert that fallback was used

# --- Tests for load_grouped_epochs ---
def test_load_grouped_epochs(load_data_instance):
    parameters = {"fixed_window_size": 10}

    # Tests if calls of dispatched functions are correct
    # Test 'awake'
    with patch.object(LoadData, 'load_awake_times_as_df') as mock_awake:
        mock_awake.return_value = pd.DataFrame({"Start": [0], "End": [10], "ResultID": [1]})
        res = load_data_instance.load_grouped_epochs(parameters, "awake")
        assert res == {1: [(0, 10)]}
        
    # Test 'faw'
    with patch.object(LoadData, 'load_faw_times_as_df') as mock_faw:
        mock_faw.return_value = pd.DataFrame({"Start": [5], "End": [15], "ResultID": [2]})
        res = load_data_instance.load_grouped_epochs(parameters, "faw")
        assert res == {2: [(5, 15)]}
        
    # Test 'normal_an'
    with patch.object(LoadData, 'sample_anesthesia_epochs') as mock_sample:
        mock_sample.return_value = pd.DataFrame({"Start": [100], "End": [110], "ResultID": [3]})
        res = load_data_instance.load_grouped_epochs(parameters, "normal_an", num_epochs=1)
        assert res == {3: [(100, 110)]}
        mock_sample.assert_called_with(parameters, num_epochs=1)

    # Test invalid
    with pytest.raises(ValueError, match="not recognized"):
        load_data_instance.load_grouped_epochs(parameters, "invalid")

# --- Tests for load_eeg_data ---
def test_load_eeg_data_filtered(load_data_instance):
    # Setup
    result_id = 1
    expected_fs = 128
    expected_eeg = np.array([[1, 2]])
    
    with patch.object(LoadData, '_load_filtered_eeg_data') as mock_filtered:
        mock_filtered.return_value = (expected_fs, expected_eeg)
        
        # Execute
        fs, eeg = load_data_instance.load_eeg_data(result_id, filtered=True)
        
        # Verify
        assert fs == expected_fs
        np.testing.assert_array_equal(eeg, expected_eeg)
        mock_filtered.assert_called_once_with(result_id)

def test_load_eeg_data_raw(load_data_instance):
    # Setup
    result_id = 2
    expected_fs = 256
    expected_eeg = np.array([[3, 4]])
    
    with patch.object(LoadData, '_load_raw_eeg_data') as mock_raw:
        mock_raw.return_value = (expected_fs, expected_eeg)
        
        # Execute
        fs, eeg = load_data_instance.load_eeg_data(result_id, filtered=False)
        
        # Verify
        assert fs == expected_fs
        np.testing.assert_array_equal(eeg, expected_eeg)
        mock_raw.assert_called_once_with(result_id)

# --- Tests for load_model ---
def test_load_model(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    model_key = "svm"
    parameters = {"h": 1}
    mock_model = MagicMock()
    
    mock_path_manager.get_complex_ml_path.return_value = tmp_path
    
    with patch('joblib.load', return_value=mock_model) as mock_joblib_load:
        # Execute
        model = load_data_instance.load_model(model_key, parameters)
        
        # Verify
        assert model == mock_model
        mock_joblib_load.assert_called_once()
        # Verify if it looked for the correct filename
        args, _ = mock_joblib_load.call_args
        assert args[0].name == f"{model_key}.joblib"

# --- Tests for load_combined_features_df ---
def test_load_combined_features_df(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"param": "val"}
    class_1 = "normal_an"
    class_0 = "awake"
    
    path_1 = tmp_path / "class1.csv"
    path_0 = tmp_path / "class0.csv"
    
    df1 = pd.DataFrame({"feat": [1, 2]})
    df0 = pd.DataFrame({"feat": [3, 4]})
    df1.to_csv(path_1, index=False)
    df0.to_csv(path_0, index=False)

    # Makes sure that the resolve_episode_path calls return path_1 and path_0 subsequently
    mock_path_manager.resolve_episode_path.side_effect = [path_1, path_0]
    
    # Execute
    res1, res0 = load_data_instance.load_combined_features_df(parameters, class_1, class_0)
    
    # Verify
    pd.testing.assert_frame_equal(res1, df1)
    pd.testing.assert_frame_equal(res0, df0)
    assert mock_path_manager.resolve_episode_path.call_count == 2

# --- Tests for load_outliers ---
def test_load_outliers_from_csv(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"h": 1}
    model_key = "svm"
    outlier_run_name = "run1"
    
    mock_path_manager.get_complex_ml_path.return_value = tmp_path
    
    # Mock CSV file
    df_outliers = pd.DataFrame({"group": [10, 20], "other": ["a", "b"]})
    file_path = tmp_path / "Summary_outliers_by_groups.csv"
    df_outliers.to_csv(file_path, index=False)
    
    # Execute
    result = load_data_instance.load_outliers(parameters, model_key, outlier_run_name, False, True)
    
    # Verify
    assert result == [10, 20]

def test_load_outliers_create_from_results(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"h": 1}
    model_key = "svm"
    
    mock_path_manager.get_complex_ml_path.return_value = tmp_path
    # No CSV file exists

    # --- Mock MetaFoldAnalyzer ---
    # 1. Create the module mock
    mock_mfa_module = MagicMock()

    # 2. Mock a MetaFoldAnalyzer instance and set the return value for its 'method select_outlier_groups'
    mock_instance = mock_mfa_module.MetaFoldAnalyzer.return_value
    mock_instance.select_outlier_groups.return_value = pd.DataFrame({"group": [30]})
    # --- Mock MetaFoldAnalyzer end ---
    
    with patch.dict('sys.modules', {
        'MachineLearning.Evaluation.meta_fold_analyzer': mock_mfa_module
    }):
        # Execute
        result = load_data_instance.load_outliers(parameters, model_key, None, False, True)

        # Verify
        assert result == [30]

def test_load_outliers_global(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"h": 1}
    model_key = "svm"
    mock_path_manager.get_complex_ml_path.return_value = tmp_path

    # Mock CSV file
    df_run_outliers = pd.DataFrame({"group": [1]})
    file_path = tmp_path / "Summary_outliers_by_groups.csv"
    df_run_outliers.to_csv(file_path, index=False)

    # Define expected global outliers
    df_global_outliers = pd.DataFrame({"group": [1, 2, 3]})
    
    # --- Mock SaveResult ---
    # 1. Create the instance mock (this is what actually gets called)
    mock_save_instance = MagicMock()

    # 2. Create the class mock that returns our instance mock
    mock_save_cls = MagicMock(return_value=mock_save_instance)

    # 3. Create a module mock and attach the class mock to it
    mock_save_module = MagicMock()
    mock_save_module.SaveResult = mock_save_cls
    # --- Mock SaveResult end ---

    # Patch sys.modules to return the custom module mock when imported
    with patch.dict('sys.modules', {
        'MachineLearning.IO.save_result': mock_save_module
    }):
        with patch.object(LoadData, 'load_global_outliers') as mock_load_global:
            mock_load_global.return_value = df_global_outliers
            # Execute
            result = load_data_instance.load_outliers(parameters, model_key, None, True, True)

            # Verify
            assert result == [1, 2, 3]
            mock_save_instance.save_global_outliers.assert_called_once_with(parameters, ANY, "patient_id")

def test_load_outliers_not_found(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"h": 1}
    mock_path_manager.get_complex_ml_path.return_value = tmp_path

    # --- Mock MetaFoldAnalyzer ---
    # 1. Create the module mock
    mock_mfa_module = MagicMock()

    # 2. Mock a MetaFoldAnalyzer instance and make its 'method select_outlier_groups' create a FileNotFoundError
    mock_instance = mock_mfa_module.MetaFoldAnalyzer.return_value
    mock_instance.select_outlier_groups = MagicMock(side_effect=FileNotFoundError)
    # --- Mock MetaFoldAnalyzer end ---

    with patch.dict('sys.modules', {
        'MachineLearning.Evaluation.meta_fold_analyzer': mock_mfa_module
    }):
        result = load_data_instance.load_outliers(parameters, "svm", None, False, True)

        # Verify
        assert result is None

# --- Tests for load_results ---
def test_load_results(load_data_instance, mock_path_manager, tmp_path):
    # Setup
    hyperparameters = MagicMock()
    run_name = MagicMock()
    model_key = MagicMock()
    
    file1 = tmp_path / "res1.csv"
    file2 = tmp_path / "res2.csv"
    pd.DataFrame({"val": [1]}).to_csv(file1, index=False)
    pd.DataFrame({"val": [2]}).to_csv(file2, index=False)
    
    mock_path_manager.get_related_paths.return_value = [file1, file2]
    
    # 1. Combined=True
    res_comb = load_data_instance.load_results(hyperparameters, run_name, model_key, combined=True)
    assert len(res_comb) == 2
    assert list(res_comb["val"]) == [1, 2]
    
    # 2. Combined=False
    res_dict = load_data_instance.load_results(hyperparameters, run_name, model_key, combined=False)
    assert isinstance(res_dict, dict)
    assert "res1" in res_dict
    assert "res2" in res_dict
    pd.testing.assert_frame_equal(res_dict["res1"], pd.DataFrame({"val": [1]}))
