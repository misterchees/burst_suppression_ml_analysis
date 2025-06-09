from MachineLearning.IO.io_core import IOCore, PathUtils
from MachineLearning.Utils.feature_utils import FeatureUtils
from MachineLearning.Preprocessing.filtering import Filtering
from MachineLearning.Features.eeg_feature_extractor import EEGFeatureExtractor


class Pipeline:
    result_ids = []
    io_core = IOCore()
    filtering = Filtering()
    features = EEGFeatureExtractor()
    feature_utils = FeatureUtils()

    def __init__(self, initial_data_key: str = "raw_eeg_mat"):
        """
        Sets subset of Patient IDs, i.e. subdirectory of initial data
        :param initial_data_key: Key for subdirectory in initial data, that contains subset of patient IDs
        """
        path_to_subdir = self.io_core.level2_subdir_path("initial_data", initial_data_key)
        self.result_ids = PathUtils.return_all_result_ids(path_to_subdir)

    def raw_eeg_filtering(self):
        """ Applies filtering to all EEGs specified by the id-list in this class"""
        self.filtering.filter_multiple_eeg(eeg_list=self.result_ids)

    def feature_extraction(self, all_features: bool, *custom_feature_args):
        """
        Extracts defined features from all EEGs in current result_ids subset.
        :param all_features: If True will extract all features implemented in FeatureExtractor, else will
        only extract features in custom_feature_dict.
        :param custom_feature_args: list of features to be extracted (List entries have to be feature keys)
        :return:
        """
        feature_functions = self.features.feature_extractors

        # Calls all feature extraction functions
        if all_features:
            for function in feature_functions.values():
                function(self.features)
        # calls all functions specified in custom_feature_dict by function keys
        else:
            # validate keys
            feature_keys = self.feature_utils.return_all_features_dict().keys()
            for key in custom_feature_args:
                if key not in feature_keys:
                    raise ValueError(f"'{key}' is no valid feature key. Valid keys are: {feature_keys}")
            for function_key in custom_feature_args:
                feature_functions[function_key](self.features)

    def set_result_ids(self, initial_data_key: str):
        """
        Sets subset of Patient IDs, i.e. subdirectory of initial data
        :param initial_data_key: Key for subdirectory in initial data, that contains subset of patient IDs
        """
        path_to_subdir = self.io_core.level2_subdir_path("initial_data", initial_data_key)
        self.result_ids = PathUtils.return_all_result_ids(path_to_subdir)
