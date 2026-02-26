from pathlib import Path

from MachineLearning.Utils.path_utils import PathUtils


class TestPathUtils:

    # --- Tests for purely logical methods (No file system needed) ---

    def test_return_A_B_C_D_path(self):
        """Tests if formatting and concatenation work correctly."""
        params = {
            "mac_threshold": 0.8,
            "bis_threshold": 70,
            "min_episode_length": 20,
            "refractory_time": 5
        }
        # mac 0.5 -> 050
        expected = Path("result_70_080_20_5")
        result = PathUtils.return_A_B_C_D_path("result", params)

        assert result == expected

    def test_return_X_Y_name(self):
        """Tests if X_Y_name formatting and concatenation work correctly."""
        params = {
            "merged_episodes": True,
            "fixed_window_size": 10,
            "overlap": 0.25
        }
        # overlap 0.25 -> 025
        expected = Path("Summary_Merged_Episodes_10_025")
        result = PathUtils.return_X_Y_name(params)

        assert result == expected

    def test_return_X_Y_name_not_merged(self):
        """Tests X_Y_name with unmerged episodes (default Summary_Episodes)."""
        params = {
            "merged_episodes": False,
            "fixed_window_size": 20,
            "overlap": 0.0
        }
        expected = Path("Summary_Episodes_20_000")
        result = PathUtils.return_X_Y_name(params)

        assert result == expected

    # --- Tests for File System methods (Using tmp_path fixture) ---

    def test_list_files_in_folder(self, tmp_path: Path):
        """Tests listing files, ignoring subdirectories, and filtering by extension."""
        # Arrange: Create a mock folder structure
        (tmp_path / "data1.csv").touch()
        (tmp_path / "data2.CSV").touch()  # Testing case-insensitivity
        (tmp_path / "info.txt").touch()

        # Create a subdirectory with a file (should be ignored)
        subdir = tmp_path / "subfolder"
        subdir.mkdir()
        (subdir / "hidden.csv").touch()

        # Act 1: Get all files (no filter, fullpaths=False)
        all_files, count_all = PathUtils.list_files_in_folder(tmp_path, fullpaths=False)

        # Assert 1
        assert count_all == 3
        assert "data1.csv" in all_files
        assert "info.txt" in all_files
        assert "subfolder" not in all_files  # Subdirectories should be ignored

        # Act 2: Filter by .csv and return fullpaths
        csv_files, count_csv = PathUtils.list_files_in_folder(tmp_path, extension_filter=".csv", fullpaths=True)

        # Assert 2
        assert count_csv == 2
        assert (tmp_path / "data1.csv") in csv_files
        assert (tmp_path / "data2.CSV") in csv_files  # Should catch uppercase .CSV too

    def test_list_files_missing_folder(self, tmp_path: Path):
        """Tests safe handling of non-existent folders."""
        fake_folder = tmp_path / "does_not_exist"
        files, count = PathUtils.list_files_in_folder(fake_folder)

        assert files == []
        assert count == 0

    def test_clear_folder(self, tmp_path: Path):
        """Tests if clear_folder deletes only files and leaves subdirectories intact."""
        # Arrange
        file_to_delete = tmp_path / "delete_me.txt"
        file_to_delete.touch()
        file_to_delete2 = tmp_path / "delete_me2.txt"
        file_to_delete.touch()
        file_to_delete3 = tmp_path / "delete_m3.csv"
        file_to_delete.touch()

        subdir = tmp_path / "keep_me_dir"
        subdir.mkdir()

        file_in_subdir = subdir / "keep_me.txt"
        file_in_subdir.touch()

        # Act
        PathUtils.clear_folder(tmp_path)

        # Assert
        assert not file_to_delete.exists(), "'delete_me.txt' in root folder should be deleted."
        assert not file_to_delete2.exists(), "'delete_me2.txt' in root folder should be deleted."
        assert not file_to_delete3.exists(), "'delete_me3.csv' in root folder should be deleted."
        assert subdir.exists(), "Subdirectory should NOT be deleted."
        assert file_in_subdir.exists(), "Files inside subdirectory should NOT be deleted."