import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import json
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
            "eeg_files": {
                "eeg_fs": "fs",
                "eeg_rawEEG": "rawEEG",
                "eeg_channels": ["1", "2"]
            },
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
        save_result_instance.save_psd_in_given_directory(np.array([]), np.array([]), None, 20, 123, tmp_path)

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

def test_save_eeg_track(save_result_instance, mock_path_manager, tmp_path):
    # Setup
    eeg_track = np.array([[0.1, 0.2], [0.3, 0.4]])
    fs = 100
    result_id = 1
    folder_keys = ["raw_data", "eegs"]
    mock_path_manager.get_path.return_value = tmp_path
    
    # Execute
    save_result_instance.save_eeg_track(eeg_track, fs, result_id, folder_keys)
    
    # Verify
    expected_file = tmp_path / "1.csv"
    assert expected_file.exists()
    
    with open(expected_file, "r") as f:
        lines = f.readlines()
        assert lines[0] == "# fs = 100\n"
        # Check if data is present (header + 2 rows)
        assert len(lines) == 4
    
    df = pd.read_csv(expected_file, comment='#')
    assert list(df.columns) == ["1", "2"]
    assert len(df) == 2
    mock_path_manager.get_path.assert_called_with(*folder_keys)

def test_save_combined_features(save_result_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"param": "val"}
    merged_df = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
    epoch_type = "normal_an"
    fullpath = tmp_path / "combined_features.csv"
    mock_path_manager.resolve_episode_path.return_value = fullpath
    
    # Execute
    save_result_instance.save_combined_features(parameters, merged_df, epoch_type)
    
    # Verify
    assert fullpath.exists()
    df = pd.read_csv(fullpath)
    assert len(df) == 2
    mock_path_manager.resolve_episode_path.assert_called_with(
        parameters, epoch_type, ["test_and_train_data", "feature_sets"], True, True
    )

def test_save_single_split(save_result_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"param": "val"}
    train_df = pd.DataFrame({"a": [1]})
    test_df = pd.DataFrame({"a": [2]})
    train_test_tuple = (train_df, test_df)
    
    mock_path_manager.get_complex_ml_path.return_value = tmp_path
    
    # Execute
    save_result_instance.save_single_split(parameters, train_test_tuple)
    
    # Verify
    assert (tmp_path / "train_split.csv").exists()
    assert (tmp_path / "test_split.csv").exists()
    mock_path_manager.get_complex_ml_path.assert_called_with(
        parameters, ["test_and_train_data", "splits"], False, True
    )

def test_save_cv_splits_to_csv(save_result_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"param": "val"}
    X = pd.DataFrame({"feat1": [11, 12, 13, 14, 21, 22, 23, 24], "feat2": [11, 12, 13, 14, 21, 22, 23, 24]}) # 2x8 feature matrix
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1]) # 1x8 label vector
    # 2 folds; Value range 0-7; Fold tupels: (train_indices, test_indices)
    fold1 = (np.array([0, 1, 2]), np.array([2, 3]))
    fold2 = (np.array([4, 5]), np.array([6, 7, 3 ,5]))
    splits = [fold1, fold2]

    split_object = (X, y, splits)
    
    mock_path_manager.get_complex_ml_path.return_value = tmp_path
    
    # Execute
    save_result_instance.save_cv_splits_to_csv(parameters, split_object)
    
    # Verify
    assert (tmp_path / "1_2_train_split.csv").exists()
    assert (tmp_path / "1_2_test_split.csv").exists()
    assert (tmp_path / "2_2_train_split.csv").exists()
    assert (tmp_path / "2_2_test_split.csv").exists()
    
    train_df1 = pd.read_csv(tmp_path / "1_2_train_split.csv")
    assert "label" in train_df1.columns
    assert len(train_df1) == 3

    test_df2 = pd.read_csv(tmp_path / "2_2_test_split.csv")
    assert "label" in test_df2.columns
    assert len(test_df2) == 4

def test_save_model(save_result_instance, mock_path_manager, tmp_path):
    # Setup
    model = MagicMock()
    model_key = "test_model"
    parameters = {"param": "val"}
    mock_path_manager.get_complex_ml_path.return_value = tmp_path
    
    with patch('joblib.dump') as mock_dump:
        # Execute
        save_result_instance.save_model(model, model_key, parameters)
        
        # Verify
        mock_dump.assert_called_once()
        args, _ = mock_dump.call_args
        assert args[0] == model
        assert str(args[1]).endswith("test_model.joblib")

def test_save_predicted_set(save_result_instance, mock_path_manager, tmp_path):
    # Setup
    test_df = pd.DataFrame({"label": [1, 0, 1]})
    test_path = Path("some/path/derived_name_for_final_file.csv")
    pred_df = pd.Series([1, 1, 0])
    parameters = {"param": "val"}
    model_key = "test_model"
    
    mock_path_manager.get_complex_ml_path.return_value = tmp_path
    
    # Execute
    save_result_instance.save_predicted_set(test_df, test_path, pred_df, parameters, model_key)
    
    # Verify
    expected_file = tmp_path / "derived_name_for_final_file_full_and_pred.csv"
    assert expected_file.exists()
    df = pd.read_csv(expected_file)
    assert "prediction" in df.columns
    assert "error" in df.columns
    # error should be [0, 1, 1] (where 1 means mismatch)
    assert df["error"].tolist() == [0, 1, 1]

def test_save_global_outliers(save_result_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"param": "val"}
    outliers_df = pd.DataFrame({"id": [1, 2]})
    mock_path_manager.get_complex_ml_path.return_value = tmp_path
    
    with patch('MachineLearning.IO.save_result.FileDataUtils.append_unique_rows_to_csv') as mock_append:
        mock_append.return_value = (outliers_df, 2)
        
        # Execute & Verify for 'epoch'
        save_result_instance.save_global_outliers(parameters, outliers_df, "epoch")
        mock_append.assert_called_with(outliers_df, tmp_path / "global_epoch_outliers.csv")
        
        # Execute & Verify for 'patient_id'
        save_result_instance.save_global_outliers(parameters, outliers_df, "patient_id")
        mock_append.assert_called_with(outliers_df, tmp_path / "global_patient_outliers.csv")
        
        # Invalid type
        with pytest.raises(ValueError, match="Invalid outlier type"):
            save_result_instance.save_global_outliers(parameters, outliers_df, "invalid")

def test_save_file_unsupported_type(save_result_instance, tmp_path):
    with pytest.raises(ValueError, match="Unknown file type"):
        save_result_instance.save_file("unsupported", tmp_path, "prefix", "suffix", {})

def test_save_file_all_types(save_result_instance, tmp_path):
    # Test dataframe
    df = pd.DataFrame({"a": [1]})
    save_result_instance.save_file("dataframe", tmp_path, "df", "test", df)
    assert (tmp_path / "df_test.csv").exists()
    
    # Test dict
    d = {"b": 2}
    save_result_instance.save_file("dict", tmp_path, "dict", "test", d)
    assert (tmp_path / "dict_test.json").exists()
    
    # Test plot
    fig = plt.figure()
    save_result_instance.save_file("plot", tmp_path, "plot", "test", fig)
    assert (tmp_path / "plot_test.png").exists()
    plt.close(fig)

def test_save_plots(save_result_instance, mock_path_manager, tmp_path):
    # Setup
    parameters = {"param": "val"}
    analysis_key = "analysis"
    mock_path_manager.get_complex_ml_path.return_value = tmp_path
    
    fig1 = MagicMock()
    ax1 = MagicMock()
    fig2 = MagicMock()
    ax2 = MagicMock()

    # Execute with invalid type
    with pytest.raises(ValueError, match="figs_and_axes must be a tuple or a list of tuples"):
        save_result_instance.save_plots(parameters, analysis_key, "invalid_type(string)", "title")

    # Mock save_file to avoid actual plot saving
    with patch.object(SaveResult, 'save_file') as mock_save_file:
        # Execute saving of multiple plots
        save_result_instance.save_plots(parameters, analysis_key, [(fig1, ax1), (fig2, ax2)], "title")
        assert mock_save_file.call_count == 2
        
        mock_save_file.reset_mock()
        
        # Execute saving of single plot
        save_result_instance.save_plots(parameters, analysis_key, (fig1, ax1), "title")
        assert mock_save_file.call_count == 1

def test_save_single_plot_static(tmp_path):
    fig = plt.figure()
    fullpath = tmp_path / "test_plot.png"
    
    # Execute
    SaveResult._save_single_plot(fig, str(fullpath))
    
    # Verify
    assert fullpath.exists()
    
    # Test fig is None
    SaveResult._save_single_plot(None, str(fullpath)) # Should just print

def test_save_file_as_csv_static(tmp_path):
    df = pd.DataFrame({"a": [1]})
    fullpath = tmp_path / "test.csv"
    
    # Execute
    SaveResult._save_file_as_csv(df, fullpath)
    
    # Verify
    assert fullpath.exists()
    assert pd.read_csv(fullpath)["a"].iloc[0] == 1


def test_save_data_as_json(save_result_instance, tmp_path):
    # Setup
    data_dict = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
    filename = "test_data.json"
    file_path = tmp_path / filename

    # Execute
    save_result_instance.save_data_as_json(data_dict, file_path)

    # Verify
    assert file_path.exists()

    with open(file_path, 'r') as f:
        loaded_data = json.load(f)

    assert loaded_data == data_dict
    assert loaded_data["key1"] == "value1"
    assert loaded_data["key2"] == 123
    assert loaded_data["key3"] == [1, 2, 3]