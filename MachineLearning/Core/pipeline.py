import pandas as pd
from fontTools.misc.classifyTools import Classifier

from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Utils.feature_utils import FeatureUtils
from MachineLearning.Features.transforms import Transforms
from MachineLearning.Features.eeg_feature_extractor import EEGFeatureExtractor
from MachineLearning.Utils.config_loader import load_config
from MachineLearning.Utils.path_utils import PathUtils


class Pipeline:
    result_ids = []

    def __init__(self, initial_data_key:str, epoch_classes: dict):
        """
        Sets subset of Patient IDs, i.e., subdirectory of initial data
        :param initial_data_key: Key for subdirectory in initial data, that contains a subset of patient IDs
        :param epoch_classes: A dict with two classes (keys 0 and 1), which will be handled throughout the pipeline.
         Valid values are: "awake", "faw" and "normal_an"
        """
        loader = LoadData()

        self.class_0 = epoch_classes[0]
        self.class_1 = epoch_classes[1]
        self.feature_extractor = EEGFeatureExtractor(*epoch_classes.values())
        self.transformer = Transforms(*epoch_classes.values())
        self.result_ids = loader.return_all_result_ids(initial_data_key)

    def raw_eeg_filtering(self):
        """ Applies filtering to all EEGs specified by the id-list in this class"""
        from MachineLearning.Preprocessing.filtering import Filtering
        filtering = Filtering()
        filtering.filter_multiple_eeg(eeg_list=self.result_ids)

    def transform_eeg_to_psd(self, channel=1, nperseg_seconds=2):
        """Wrapper for transform function implemented in Transforms class"""
        self.transformer.transform_eeg_episodes_to_psd(channel, nperseg_seconds)

    def feature_extraction(self, all_features: bool, features: list):
        """
        Extracts defined features from all EEGs in the current result_ids subset.
        :param all_features: If True extracts all features implemented in FeatureExtractor, else will
        only extract features in custom_feature_dict.
        :param features: List of features to be extracted (List entries have to be feature keys)
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
            for key in features:
                if key not in feature_keys:
                    raise ValueError(f"'{key}' is no valid feature key. Valid keys are: {feature_keys}")
            for function_key in features:
                feature_functions[function_key](self.feature_extractor)

    def combine_features(self, all_features: bool, features: list):
        """Wrapper for combining features method implemented in FeatureExtractor"""
        self.feature_extractor.combine_features(all_features, features)

    def create_splits(self, test_size: float, random_state: int, split_paths=True, folds=True, iterations: int = None):
        """
        Loads the test set, creates splits, splitting first on patient level and then tries to create equivalent
        ratios of faw and awake class in both test and train.

        :param test_size: Float that determines the ratio of test to train. E.g., 0.15 -> test:15% train:85%
        :param random_state: Randomness seed, to reproduce the shuffling of the splits.
        :param split_paths: If True, this method returns a tuple of paths leading to split train and test files.
        :param folds: If True, the splits will be as many non-overlapping folds as possible for cross-validation.
        :param iterations: Number of iterations for searching folds. Will be ignored if param "folds" is False.
        :return: If split_paths is True, returns the split paths: (<train set path>, <test set path>).
         Depending on folds, if it is true, a list of tuples will be returned, else a single tuple will be returned.
        """
        from MachineLearning.Evaluation.split_manager import SplitManager

        parameters = self.get_current_parameters()
        split_manager = SplitManager(parameters, self.class_0, self.class_1, test_size, random_state)
        split_manager.load_and_validate()

        # create single split or folds
        if folds:
            split_manager.create_custom_splits_by_test_size(min_iterations=iterations)
            return_splits = split_manager.return_k_fold_split_paths
        else:
            split_manager.create_single_split()
            return_splits = split_manager.return_split_paths

        if split_paths:
            return return_splits()
        return None

    def run_svm_classifier(self, train_path: str, test_path: str,
                           classifier: Classifier = None, save_clf=True, save_pred=True, **kwargs):
        """
        Runs SVM classifier on train and test sets of given paths. It takes a pretrained Classifier or trains
        the base model if None is given.
        :param train_path: Fullpath to train set as string.
        :param test_path: Fullpath to test set as string.
        :param classifier: Already trained SVM Classifier.
        :param save_clf: If true will save the trained SVM classifier.
        :param save_pred: If true will save the test set with the predictions.
        :return: Tuple -> (predicted values, test labels, probabilities)
        """

        parameters = self.get_current_parameters()
        svm_key = "svm"
        test_df = pd.read_csv(test_path)

        # Setup model
        if classifier is not None:
            loader = LoadData()
            clf = loader.load_model(svm_key, parameters)  # load pretrained model
        else:
            from MachineLearning.Models.svm_classifier import SVMClassifier  # Lazy import
            # Prepare features/labels
            train_df = pd.read_csv(train_path)
            X_train = train_df.drop(columns=["Start", "End", "ResultID", "label"]).values
            y_train = train_df["label"].values

            # Train model
            clf = SVMClassifier(probability=True, **kwargs)
            clf.train(X_train, y_train)

        X_test = test_df.drop(columns=["Start", "End", "ResultID", "label"]).values
        y_test = test_df["label"].values

        # Predict
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        # Save the trained model
        if save_clf:
            saver = SaveResult()
            saver.save_model(clf, svm_key, parameters)

        # Save the original data with the prediction and error column
        if save_pred:
            saver = SaveResult()
            saver.save_predicted_set(test_df, test_path, y_pred, parameters, svm_key)

        return y_pred, y_test, y_proba

    def evaluate_metrics(self, y_test, y_pred, y_proba, folds:bool, print_metrics=True, save_metrics=True):
        """
        Wrapper for the metric evaluator of Machine Learning algorithm.
        :param y_test: Test labels which contain ground truth.
        :param y_pred: Predicted labels.
        :param y_proba: Prediction probabilities (optional, for AUC).
        :param folds: Defines the name of the prefix for the saved metrics.
        :param print_metrics: If True, prints the results of the evaluation.
        :param save_metrics: If True, saves the results of the evaluation.
        :return: A dict with the result of the evaluation.
        """
        from MachineLearning.Evaluation.metrics_evaluator import MetricsEvaluator

        evaluator = MetricsEvaluator(self.class_0, self.class_1, y_test, y_pred, y_proba)
        evaluation = evaluator.evaluate(print_metrics)  # evaluate and print results
        if save_metrics:
            saver = SaveResult()
            prefix = "folds" if folds else "single"
            saver.save_ml_result(evaluation, "svm", self.get_current_parameters(), "dict", prefix, "metrics")

        return evaluation

    def split_classify_evaluate(self, test_size: float, random_state: int, folds=True, **kwargs):
        """
        Splits data, performs classification, and evaluates the results using a specified test size,
        random state, and optionally in a cross-validation setting.

        :param test_size: Fraction of the dataset to include in the test split. Must be a float between 0 and 1.
        :param random_state: Random seed to ensure reproducibility of the data split.
        :param folds: Indicates whether to perform classification using cross-validation.
                      If True, it applies cross-validation; if False, a single split is used.
        :param kwargs: Additional optional parameters to pass to the classifier or related methods.
        :return: None
        """
        iterations = int(1//test_size)*2  # Double the number of minimal necessary iterations
        split_paths = self.create_splits(test_size, random_state, folds=folds, iterations=iterations)

        if not folds:
            train_path, test_path = split_paths
            y_pred, y_test, y_proba = self.run_svm_classifier(train_path, test_path, save_clf=False, **kwargs)
            self.evaluate_metrics(y_test, y_pred, y_proba, folds)

        else:
            y_pred, y_test, y_proba = self.collect_classification_results(split_paths, **kwargs)
            self.evaluate_metrics(y_test, y_pred, y_proba, folds)

    def set_result_ids(self, initial_data_key: str):
        """
        Sets subset of Patient IDs, i.e. subdirectory of initial data
        :param initial_data_key: Key for subdirectory in initial data, that contains subset of patient IDs
        """
        loader = LoadData()
        self.result_ids = loader.return_all_result_ids(initial_data_key)

    @staticmethod
    def get_current_parameters() -> dict:
        """Returns the current parameters as a dictionary."""
        # Return current parameters stored in config
        return load_config("parameters_config.yaml")["current_params"]

    def collect_classification_results(self, split_paths: list, **kwargs) -> tuple[list, list, list]:
        """
        Collects the classification results from multiple splits.
        Args:
            split_paths: List of tuples, each containing the paths to the train and test set of a split.
            kwargs: kwargs for the SVM classifier

        Returns:
            tuple: (predictions, true_labels, probabilities)
        """
        predictions = []
        true_labels = []
        probabilities = []

        for train_path, test_path in split_paths:
            prediction, true_label, probability = self.run_svm_classifier(
                train_path,
                test_path,
                save_clf=False,
                **kwargs
            )
            predictions.append(prediction)
            true_labels.append(true_label)
            probabilities.append(probability)

        return predictions, true_labels, probabilities

    def analyze_single_result(self, result_path: str, metadata_to_test: str, print_analysis=True, save_analysis=True):
        """
        Analyzes a single result file to evaluate the correlation between metadata and error rates, categorize errors by
        metadata groups, and assess the distribution of classes and confusion matrices for specified metadata.

        :param result_path: The file path to the CSV containing the results to be analyzed.
        :type result_path: str
        :param metadata_to_test: The name of the metadata column in the results CSV to be analyzed.
        :type metadata_to_test: str
        :param print_analysis: A flag indicating whether the analysis results should be printed to the console.
        :type print_analysis: bool, optional
        :param save_analysis: A flag indicating whether the analysis results should be saved to disk.
        :type save_analysis: bool, optional
        :return: None
        """
        from MachineLearning.Evaluation.metadata_analyzer import MetadataAnalyzer

        print(f"Analyzing result file: {result_path}")

        # Read results from path and verify if metadata is a valid column name
        result_df = pd.read_csv(result_path)
        if metadata_to_test not in result_df.columns:
            raise ValueError(f"Metadata column '{metadata_to_test}' not found in result file.")

        analyzer = MetadataAnalyzer(result_df)

        # Calculate analysis
        error_correlation = analyzer.correlation_with_error()
        error_by_metadata = analyzer.error_by_group(metadata_to_test)
        class_dist_per_metadata = analyzer.class_distribution_by_group(metadata_to_test)
        confusion_matrices_by_metadata = analyzer.confusion_matrix_by_group(metadata_to_test)

        if print_analysis:
            print(f"Correlation with error: {error_correlation}")
            print(f"Error by {metadata_to_test}: {error_by_metadata}")
            print(f"Class distribution by {metadata_to_test}: {class_dist_per_metadata}")
            print(f"Confusion matrices by {metadata_to_test}: {confusion_matrices_by_metadata}")

        # create plots
        error_dist_by_metadata = analyzer.plot_error_distribution(metadata_to_test, print_analysis)
        temp_error_by_metadata = analyzer.plot_temporal_error(metadata_to_test, print_analysis)

        if save_analysis:
            print(f"Saving analysis results to disk...")
            filename = PathUtils.return_filename_from_fullpath(result_path)
            saver = SaveResult()
            saver.save_ml_result(error_correlation, "svm", self.get_current_parameters(),
                                 "dataframe", filename, "error_correlation")

            saver.save_ml_result(error_by_metadata, "svm", self.get_current_parameters(),
                                 "dataframe", filename, f"error_by_{metadata_to_test}")

            saver.save_ml_result(class_dist_per_metadata, "svm", self.get_current_parameters(),
                                 "dataframe", filename, f"class_dist_per_{metadata_to_test}")

            saver.save_ml_result(confusion_matrices_by_metadata, "svm", self.get_current_parameters(),
                                 "dict", filename, f"confusion_matrices_by_{metadata_to_test}")

            saver.save_ml_result(error_dist_by_metadata, "svm", self.get_current_parameters(),
                                 "plot", filename, f"error_dist_by_{metadata_to_test}")

            saver.save_ml_result(temp_error_by_metadata, "svm", self.get_current_parameters(),
                                 "plot", filename, f"temp_error_by_{metadata_to_test}")


        print("Analysis complete.")

    def analyze_results(self, model_key: str, metadata_to_test: list, print_analysis=True, save_analysis=True):

        # Gather all results
        loader = LoadData()
        parameter_dict = self.get_current_parameters()
        result_folder = loader.return_all_parameter_fullpath(parameter_dict,False, False, "results", model_key)
        path_list,_ = PathUtils.list_files_in_folder(result_folder, ".csv", fullpaths=True)

        # Analyze all given metadata in all results
        for metadata in metadata_to_test:
            for result_path in path_list:
                self.analyze_single_result(result_path, metadata, print_analysis, save_analysis)
