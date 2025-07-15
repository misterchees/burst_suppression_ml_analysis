import warnings
import pandas as pd
from fontTools.misc.classifyTools import Classifier

from MachineLearning.IO.load_data import LoadData, PathUtils
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Features.transforms import Transforms
from MachineLearning.Features.eeg_feature_extractor import EEGFeatureExtractor
from MachineLearning.Utils.config_handler import load_config, update_config
from MachineLearning.Evaluation.comparison import Comparison
from MachineLearning.Utils.feature_utils import FeatureUtils


class Pipeline:
    result_ids = []

    def __init__(self, init_data_key: str, epoch_classes: dict, parameters: dict = None, features: list | str = None):
        """
        Sets subset of Patient IDs, i.e., subdirectory of initial data
        :param init_data_key: Key for subdirectory in initial data, that contains a subset of patient IDs
        :param epoch_classes: A dict with two classes (keys 0 and 1), which will be handled throughout the pipeline.
         Valid values are: "awake", "faw" and "normal_an"
        :param parameters: Parameters for the pipeline. If None, the current parameters will be used.
        :param features: List of features to use in the pipeline. If all existing features should be used, also a string
                with value "all_features" can be passed.
        """
        loader = LoadData()

        self.class_0 = epoch_classes[0]
        self.class_1 = epoch_classes[1]
        self.features = features

        self.all_features = self._check_features()  # Flag to determine which features to handle

        self.update_parameters(parameters)  # Update current parameters with given parameters

        if self.all_features is None:
            self.transforms = None
            self.feature_extractor = None
        else:
            # Check for each epoch if calculations can be skipped
            epochs_tuple = tuple(epoch_classes.values())
            transform_epochs = []  # list with transform epochs
            feature_epochs = []  # list with feature extraction epochs
            for epoch_type in epochs_tuple:
                if not self.already_calculated("transform", epoch_type):
                    transform_epochs.append(epoch_type)
                if not self.already_calculated("extract_features", epoch_type):
                    feature_epochs.append(epoch_type)

            if not transform_epochs:
                self.transformer = None
            else:
                self.transformer = Transforms(tuple(transform_epochs), parameters)
            if not feature_epochs:
                self.feature_extractor = None
            else:
                self.feature_extractor = EEGFeatureExtractor(tuple(feature_epochs), parameters)
        self.result_ids = loader.return_all_result_ids(init_data_key)

    def already_calculated(self, calculation_type: str, epoch_type: str) -> bool:
        """
        Checks if calculation type was already performed on given epoch type.
        :param calculation_type: The type of calculation. Allowed values are "extract_features" and "transform"
        :param epoch_type: Epoch type. Allowed values are "awake", "faw" and "normal_an".
        :return: True if calculation type was already performed on given epoch type, else False.
        """

        # load faw and awake data dependent of epochtypes
        loader = LoadData()
        parameters = self.get_current_parameters()

        if epoch_type == "normal_an":
            return False
        elif epoch_type == "awake":
            times_df = loader.load_awake_times_as_df(parameters)
        elif epoch_type == "faw":
            times_df = loader.load_faw_times_as_df(parameters)
        else:
            raise ValueError("Epoch type must be 'normal_an', 'awake' or 'faw'")

        if calculation_type == 'transform':
            psd_folderpath = loader.return_file_fullpath(parameters, False, False, epoch_type, "features", "psds")
            comparison_dict = Comparison.compare_csv_to_psd_folder(times_df, psd_folderpath)
            identical = bool(comparison_dict["a_in_b"] and comparison_dict["b_in_a"])
            return identical

        elif calculation_type == 'extract_features':
            if self.all_features:
                feature_list = list(self.feature_extractor.feature_extract_funcs.keys())
            else:
                feature_list = self.features

            # Returns True if ALL features are already calculated, False otherwise
            for feature in feature_list:
                feature_filepath = loader.return_file_fullpath(parameters, True, False, epoch_type, "features", feature)
                comparison_dict = Comparison.compare_two_csv(feature_filepath, times_df)
                if not (comparison_dict["a_in_b"] and comparison_dict["b_in_a"]):
                    return False
            return True
        else:
            raise ValueError("Calculation type must be 'transform' or 'extract_features'")

    def raw_eeg_filtering(self):
        """ Applies filtering to all EEGs specified by the id-list in this class"""
        from MachineLearning.Preprocessing.filtering import Filtering
        filtering = Filtering()
        filtering.filter_multiple_eeg(eeg_list=self.result_ids)

    def transform_eeg_to_psd(self, channel=1, nperseg_seconds=2):
        """Wrapper for transform function implemented in Transforms class"""
        if self.transformer is None:
            print("Skipping PSD transforms")
            return
        self.transformer.transform_eeg_episodes_to_psd(channel, nperseg_seconds)

    def feature_extraction(self):
        """
        Extracts defined features from all EEGs in the current result_ids subset. Depending on the features
        given to this pipeline instance it will extract features.
        """
        if self.all_features is None or self.feature_extractor is None:
            print("Skipping feature extraction")
            return

        feature_functions = self.feature_extractor.feature_extract_funcs

        # Calls all feature extraction functions
        if self.all_features:
            for function in feature_functions.values():
                function(self.feature_extractor)

        # calls all functions implemented in feature extractor
        else:
            for function_key in self.features:
                feature_functions[function_key](self.feature_extractor)

    def combine_features(self, features: list = None):
        """Wrapper for combining features method implemented in FeatureExtractor"""
        if features is None:
            features = self.features
        if self.feature_extractor is None:
            print("Skipping feature combination")
            return
        self.feature_extractor.combine_features(self.all_features, features)

    def _check_features(self) -> bool | None:
        """
        Checks self.features and returns None, True or False based on the value of self.features.
        None -> No features to extract.
        True -> All features to extract.
        False -> Features from given features to extract.
        """
        # skip if features are None or empty list
        if self.features is None or not self.features:
            warnings.warn("No features were given -> No transforms or feature operations will be performed")
            return None

        # Setting all_features flag depending on value of features
        if self.features == "all_features":
            return True
        elif not isinstance(self.features, list):
            raise ValueError("Features must be either a list or a string with value 'all_features'")
        else:
            known_features = FeatureUtils.return_all_features_dict()
            # validate features given in list
            feature_keys = known_features.keys()
            for key in self.features:
                if key not in feature_keys:
                    raise ValueError(f"'{key}' is no valid feature key. Valid keys are: {feature_keys}")
            return False

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

    def evaluate_metrics(self, y_test, y_pred, y_proba, folds: bool, print_metrics=True, save_metrics=True):
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
        iterations = int(1 // test_size) * 2  # Double the number of minimal necessary iterations
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
        return load_config("parameters_config.yaml")["current_params"]

    @staticmethod
    def update_parameters(updated_parameters: dict) -> dict:
        """Updates the parameters stored in config i.e., globally and returns the updated parameters as a dictionary."""
        return update_config("parameters_config.yaml", updated_parameters)["current_params"]

    @staticmethod
    def reset_parameters():
        """Resets the parameters stored in config i.e., globally."""
        default_params = load_config("parameters_config.yaml")["initial_params"]
        return update_config("parameters_config.yaml", {"current_params": default_params})

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

    def analyze_single_result(self, result_path: str, metadata_col: str, label:int, print_analysis=True, save_analysis=True,
                              plots=True):
        """
        Analyzes a single result file to evaluate the correlation between metadata and error rates, categorize errors by
        metadata groups, and assess the distribution of classes and confusion matrices for specified metadata.

        :param result_path: The file path to the CSV containing the results to be analyzed.
        :type result_path: str
        :param metadata_col: The name of the metadata column in the results CSV to be analyzed.
        :type metadata_col: str
        :param label: The label of the metadata column to be analyzed.
        :type label: int
        :param print_analysis: A flag indicating whether the analysis results should be printed to the console.
        :type print_analysis: bool, optional
        :param save_analysis: A flag indicating whether the analysis results should be saved to disk.
        :type save_analysis: bool, optional
        :param plots: A flag indicating whether plots should be generated for the analysis results.
        :type plots: bool, optional
        :return: None
        """
        from MachineLearning.Evaluation.metadata_analyzer import MetadataAnalyzer

        print(f"Analyzing result file: {result_path}")

        # Read results from path and verify if metadata is a valid column name
        result_df = pd.read_csv(result_path)
        if metadata_col not in result_df.columns:
            raise ValueError(f"Metadata column '{metadata_col}' not found in result file.")

        analyzer = MetadataAnalyzer(result_df)

        # Calculate analysis
        error_correlation = analyzer.correlation_with_error()
        error_by_metadata = analyzer.error_by_group(metadata_col)
        label_error_by_metadata = analyzer.error_for_label_by_group(metadata_col, label)
        class_dist_per_metadata = analyzer.class_distribution_by_group(metadata_col)
        confusion_matrices_by_metadata = analyzer.confusion_matrix_by_group(metadata_col)

        if print_analysis:
            print(f"Correlation with error: {error_correlation}")
            print(f"Error by {metadata_col}: {error_by_metadata}")
            print(f"Error for label {label} by {metadata_col}: {label_error_by_metadata}")
            print(f"Class distribution by {metadata_col}: {class_dist_per_metadata}")
            print(f"Confusion matrices by {metadata_col}: {confusion_matrices_by_metadata}")

        # create plots if needed
        if plots:
            error_dist_by_metadata = analyzer.plot_error_distribution(metadata_col, print_analysis)
            temp_error_by_metadata = analyzer.plot_temporal_error(metadata_col, print_analysis)

        if save_analysis:
            print(f"Saving analysis results to disk...")
            filename = PathUtils.return_filename_from_fullpath(result_path)
            saver = SaveResult()
            saver.save_metadata_analysis(error_correlation, "svm", self.get_current_parameters(),
                                         "dataframe", filename, "error_correlation")

            saver.save_metadata_analysis(error_by_metadata, "svm", self.get_current_parameters(),
                                         "dataframe", filename, f"error_by_{metadata_col}")

            saver.save_metadata_analysis(error_by_metadata, "svm", self.get_current_parameters(),
                                         "dataframe", filename, f"error_label_{label}_by_{metadata_col}")

            saver.save_metadata_analysis(class_dist_per_metadata, "svm", self.get_current_parameters(),
                                         "dataframe", filename, f"class_dist_per_{metadata_col}")

            saver.save_metadata_analysis(confusion_matrices_by_metadata, "svm", self.get_current_parameters(),
                                         "dict", filename, f"confusion_matrices_by_{metadata_col}")

            if plots:
                saver.save_metadata_analysis(error_dist_by_metadata, "svm", self.get_current_parameters(),
                                             "plot", filename, f"error_dist_by_{metadata_col}")

                saver.save_metadata_analysis(temp_error_by_metadata, "svm", self.get_current_parameters(),
                                             "plot", filename, f"temp_error_by_{metadata_col}")

        print("Analysis complete.")

    def analyze_meta_analyses(self, model_key: str, metadata_col, print_analysis=True, save_analysis=True, plots=True):

        from MachineLearning.Evaluation.meta_fold_analyzer import MetaFoldAnalyzer

        print("Analyzing single fold analysis results")

        # Carry out analysis
        parameters = self.get_current_parameters()
        fold_analyzer = MetaFoldAnalyzer(model_key, parameters)
        fold_analyzer.load_all_folds(metadata_col)
        agg_err_by_group = fold_analyzer.aggregate_error_by_group()
        agg_label_err_by_group = fold_analyzer.aggregate_error_by_group(True)
        acc_vs_class_dist = fold_analyzer.analyze_class_imbalance_vs_metric("accuracy")
        prec_vs_class_dist = fold_analyzer.analyze_class_imbalance_vs_metric("precision")
        rec_vs_class_dist = fold_analyzer.analyze_class_imbalance_vs_metric("recall")

        if print_analysis:
            print(f"Aggregated Error by {metadata_col}: {agg_err_by_group}")
            print(f"Aggregated Error for one label by {metadata_col}: {agg_label_err_by_group}")
            print(f"Accuracy vs class distribution: {acc_vs_class_dist}")
            print(f"Precision vs class distribution: {prec_vs_class_dist}")
            print(f"Recall vs class distribution: {rec_vs_class_dist}")

        if plots:
            error_heatmap = fold_analyzer.plot_foldwise_error_heatmap(metadata_col, print_analysis)

        if save_analysis:
            saver = SaveResult()
            saver.save_metadata_analysis(agg_err_by_group, "svm", parameters, "dataframe", "Summary_analysis",
                                         "agg_error_by_groups")
            saver.save_metadata_analysis(agg_label_err_by_group, "svm", parameters, "dataframe", "Summary_analysis",
                                         "agg_label_error_by_groups")
            saver.save_metadata_analysis(acc_vs_class_dist, "svm", parameters, "dataframe", "Summary_analysis",
                                         "acc_vs_class_distribution")
            saver.save_metadata_analysis(prec_vs_class_dist, "svm", parameters, "dataframe", "Summary_analysis",
                                         "prec_vs_class_distribution")
            saver.save_metadata_analysis(rec_vs_class_dist, "svm", parameters, "dataframe", "Summary_analysis",
                                         "rec_vs_class_distribution")
            if plots:
                saver.save_metadata_analysis(error_heatmap, "svm", parameters, "plot", "Summary_analysis",
                                             "error_heatmap")

        print("Analysis of single analysis results complete")

    def analyze_results(self, model_key: str, metadata_list: list, print_analysis=True, save_analysis=True, plots=True):

        # Gather all results
        loader = LoadData()
        parameter_dict = self.get_current_parameters()
        result_folder = loader.return_all_parameter_fullpath(parameter_dict, False, False, "results", model_key)
        path_list, _ = PathUtils.list_files_in_folder(result_folder, ".csv", fullpaths=True)

        # Analyze all given metadata in all results
        for metadata in metadata_list:
            for result_path in path_list:
                self.analyze_single_result(result_path, metadata,1, print_analysis, save_analysis, plots)

            # Summary Analysis of single analysis results
            self.analyze_meta_analyses(model_key, metadata, print_analysis, save_analysis, plots)
