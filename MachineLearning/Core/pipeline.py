import pandas as pd
from fontTools.misc.classifyTools import Classifier

from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Utils.feature_utils import FeatureUtils
from MachineLearning.Features.transforms import Transforms
from MachineLearning.Features.eeg_feature_extractor import EEGFeatureExtractor
from MachineLearning.Evaluation.split_manager import SplitManager


class Pipeline:
    result_ids = []

    def __init__(self, initial_data_key="raw_eeg_mat", *epoch_types):
        """
        Sets subset of Patient IDs, i.e. subdirectory of initial data
        :param initial_data_key: Key for subdirectory in initial data, that contains subset of patient IDs
        :param epoch_types: Defines which Epochs will be handled throughout the pipeline.
        """
        loader = LoadData()

        self.epoch_types = epoch_types
        self.feature_extractor = EEGFeatureExtractor(*epoch_types)
        self.transformer = Transforms(*epoch_types)
        self.result_ids = loader.return_all_result_ids(initial_data_key)

    def raw_eeg_filtering(self):
        """ Applies filtering to all EEGs specified by the id-list in this class"""
        from MachineLearning.Preprocessing.filtering import Filtering
        filtering = Filtering()
        filtering.filter_multiple_eeg(eeg_list=self.result_ids)

    def transform_eeg_to_psd(self, channel=1, nperseg_seconds=2):
        """Wrapper for transform function implemented in Transforms class"""
        self.transformer.transform_eeg_episodes_to_psd(channel, nperseg_seconds)

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

    def combine_features(self, all_features: bool, *features):
        """Wrapper for combining features method implemented in FeatureExtractor"""
        self.feature_extractor.combine_features(all_features, *features)

    def create_splits(self, class_0: str, class_1: str,  test_size: float, random_state: int, split_paths=True):
        """
        Loads the test set, creates splits, splitting first on patient level and then tries to create equivalent
        ratios of faw and awake class in both test and train.
        :param class_0: Determines the set of epochs to be used as class with label 0. Should be different from class_1.
        Valid options are: 'awake', 'faw' and 'normal_an'
        :param class_1: Determines the set of epochs to be used as class with label 1. Should be different from class_0.
        Valid options are: 'awake', 'faw' and 'normal_an'
        :param test_size: Float that determines the ratio of test to train. E.g. 0.15 -> test:015/train:0.85
        :param random_state: Randomness seed, to reproduce the shuffling of the splits.
        :param split_paths: If True, this method returns a tuple of paths leading to splitted train and test files.
        :return: if split_paths is True, returns: (<train set path>, <test set path>), else doesn't return anything.
        """
        parameters = self.get_current_parameters()
        split_manager = SplitManager(parameters, class_0, class_1, test_size, random_state)
        split_manager.load_and_validate()
        split_manager.create_single_split()
        split_manager.save()

        if split_paths:
            return split_manager.return_split_paths()

    def run_svm_classifier(self, train_path: str, test_path: str, classifier: Classifier = None, save_clf=True):
        """
        Runs SVM classifier on train and test sets of given paths. It takes a pretrained Classifier or trains
        the base model if None is given.
        :param train_path: Fullpath to train set as string.
        :param test_path: Fullpath to test set as string.
        :param classifier: Already trained SVM Classifier.
        :param save_clf: If true will save the trained SVM classifier.
        :return: returns a tuple -> (predicted values, test labels, probabilities)
        """

        parameters = self.get_current_parameters()
        test_df = pd.read_csv(test_path)

        if classifier is not None:
            loader = LoadData()
            clf = loader.load_model("svm", parameters)
        else:
            from MachineLearning.Models.svm_classifier import SVMClassifier  # Lazy import
            # Prepare features/labels
            train_df = pd.read_csv(train_path)
            X_train = train_df.drop(columns=["Start", "End", "ResultID", "label"]).values
            y_train = train_df["label"].values

            # Train model
            clf = SVMClassifier(probability=True)
            clf.train(X_train, y_train)

        X_test = test_df.drop(columns=["Start", "End", "ResultID", "label"]).values
        y_test = test_df["label"].values

        # Predict
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        if save_clf:
            saver = SaveResult()
            saver.save_model(clf, "svm", parameters)
        return y_pred, y_test, y_proba

    @staticmethod
    def evaluate_metrics(class_0, class_1, y_test, y_pred, y_proba):
        """
        Wrapper for the metric evaluator of Machine Learning algorithm.
        :param class_0: Class 0 name
        :param class_1: Class 1 name
        :param y_test: Test labels, contains ground truth.
        :param y_pred: Predicted labels.
        :param y_proba: Prediction probabilities (optional, for AUC).
        :return: A dict with the result of the evaluation.
        """
        from MachineLearning.Evaluation.metrics_evaluator import MetricsEvaluator

        evaluator = MetricsEvaluator(class_0, class_1, y_test, y_pred, y_proba)
        evaluation = evaluator.evaluate()
        return evaluation

    def set_result_ids(self, initial_data_key: str):
        """
        Sets subset of Patient IDs, i.e. subdirectory of initial data
        :param initial_data_key: Key for subdirectory in initial data, that contains subset of patient IDs
        """
        loader = LoadData()
        self.result_ids = loader.return_all_result_ids(initial_data_key)

    def get_current_parameters(self) -> dict:
        """Returns the current parameters as a dictionary."""
        # It doesn't matter if taken from feature extractor or filterer
        return self.feature_extractor.parameter_dict
