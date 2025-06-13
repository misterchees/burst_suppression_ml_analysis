from MachineLearning.IO.io_core import IOCore, PathUtils
from MachineLearning.Utils.feature_utils import FeatureUtils
from MachineLearning.Preprocessing.filtering import Filtering
from MachineLearning.Features.transforms import Transforms
from MachineLearning.Features.eeg_feature_extractor import EEGFeatureExtractor


class Pipeline:
    result_ids = []

    def __init__(self, initial_data_key="raw_eeg_mat", faw=True, awake=True):
        """
        Sets subset of Patient IDs, i.e. subdirectory of initial data
        :param initial_data_key: Key for subdirectory in initial data, that contains subset of patient IDs
        :param faw: Flag indicating if pipeline will handle fake awake episodes
        :param awake: Flag indicating if pipeline will handle awake episodes
        """
        io_core = IOCore()
        self.feature_extractor = EEGFeatureExtractor(faw, awake)
        self.transformer = Transforms(faw, awake)
        path_to_subdir = io_core.level2_subdir_path("initial_data", initial_data_key)
        self.result_ids = PathUtils.return_all_result_ids(path_to_subdir)

    def raw_eeg_filtering(self):
        """ Applies filtering to all EEGs specified by the id-list in this class"""
        filtering = Filtering()
        filtering.filter_multiple_eeg(eeg_list=self.result_ids)

    def transform_eeg_to_psd(self, channel=1, nperseg_seconds=2, filtered=True):
        self.transformer.transform_eeg_episodes_to_psd(channel, nperseg_seconds, filtered)

    def feature_extraction(self, all_features: bool, *custom_feature_args):
        """
        Extracts defined features from all EEGs in current result_ids subset.
        :param all_features: If True will extract all features implemented in FeatureExtractor, else will
        only extract features in custom_feature_dict.
        :param custom_feature_args: list of features to be extracted (List entries have to be feature keys)
        :return:
        """
        feature_functions = self.feature_extractor.feature_extract_funcs

        # Calls all feature extraction functions
        if all_features:
            for function in feature_functions.values():
                function(self.feature_extractor)
        # calls all functions specified in custom_feature_dict by function keys
        else:
            # validate keys
            feature_keys = FeatureUtils.return_all_features_dict().keys()
            for key in custom_feature_args:
                if key not in feature_keys:
                    raise ValueError(f"'{key}' is no valid feature key. Valid keys are: {feature_keys}")
            for function_key in custom_feature_args:
                feature_functions[function_key](self.feature_extractor)

    def combine_all_features(self):
        self.feature_extractor.combine_all_features()

    def combine_features(self, *features):
        self.feature_extractor.combine_features(*features)

    def set_result_ids(self, initial_data_key: str):
        """
        Sets subset of Patient IDs, i.e. subdirectory of initial data
        :param initial_data_key: Key for subdirectory in initial data, that contains subset of patient IDs
        """
        io_core = IOCore()
        path_to_subdir = io_core.level2_subdir_path("initial_data", initial_data_key)
        self.result_ids = PathUtils.return_all_result_ids(path_to_subdir)
