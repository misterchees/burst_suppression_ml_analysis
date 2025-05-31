from MachineLearning.IO.io_core import IOCore, PathUtils
from MachineLearning.Preprocessing.filtering import Filtering


class Pipeline:
    result_ids = []
    io_core = IOCore()
    filtering = Filtering()

    def __init__(self, initial_data_key: str):
        """
        Sets subset of Patient IDs
        :param initial_data_key:
        """
        path_to_subdir = self.io_core.level2_subdir_path("initial_data", initial_data_key)
        self.result_ids = PathUtils.return_all_result_ids(path_to_subdir)

    def filter_step(self):
        self.filtering.filter_multiple_eeg(eeg_list=self.result_ids)

    def set_result_ids(self, initial_data_key: str):
        path_to_subdir = self.io_core.level2_subdir_path("initial_data", initial_data_key)
        self.result_ids = PathUtils.return_all_result_ids(path_to_subdir)
