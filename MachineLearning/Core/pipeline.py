import warnings
import pandas as pd
from fontTools.misc.classifyTools import Classifier
from pathlib import Path

from MachineLearning.Core.run_metadata import RunMetadata
from MachineLearning.Utils.path_manager import PathManager
from MachineLearning.IO.load_data import LoadData, PathUtils
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Features.transforms import Transforms
from MachineLearning.Features.eeg_feature_extractor import EEGFeatureExtractor
from MachineLearning.Utils.config_handler import load_config, update_config
from MachineLearning.Evaluation.comparison import Comparison
from MachineLearning.Utils.feature_utils import FeatureUtils


class Pipeline:
    patient_ids = []

    def __init__(self, pm: PathManager, init_data_key: str, epoch_classes: dict, update_dict: dict, filter_method: str,
                 model_key: str, normalize_method: str, transform_method: str, features_dict: dict = None,
                 metadata_to_analyze: list = None, run_name: str = None, force_overwrite: bool = False,
                 global_outliers: bool = True, force_transform: bool = False, force_extract: bool = False):
        """
        Sets subset of Patient IDs, i.e., subdirectory of initial data
        :param init_data_key: Key for subdirectory in initial data, that contains a subset of patient IDs
        :param epoch_classes: A dict with two classes (keys 0 and 1), which will be handled throughout the pipeline.
         Valid values are: "awake", "faw" and "normal_an"
        :param update_dict: A dict with all relevant parameters to update them globally i.e. in the params config.
        :param filter_method: Name of filter-method to use.
        :param model_key: Name of model to use.
        :param normalize_method: Name of normalize-method to use.
        :param transform_method: Name of transform-method to use.
        :param features_dict: Dictionary containing a list of features to extract and another to combine.
        :param metadata_to_analyze: List of metadata to analyze for error patterns in the classification.
        :param run_name: Name of the run. If None, a timestamp will be used instead.
        :param force_overwrite: If True, the pipeline will be run even if the results already exist.
        :param global_outliers: If True, global outliers found in all runs will be removed.
        :param force_transform: If True, transforms will be calculated even if the results already exist.
        :param force_extract: If True, the features will be extracted and combined even if the results already exist.
        """
        # Initialize Path Utilities
        self.pm = pm
        self.loader = LoadData(self.pm)
        self.saver = SaveResult(self.pm)

        self.class_0 = epoch_classes[0]
        self.class_1 = epoch_classes[1]

        # Update parameter config and get updated params
        updated_params = self._update_param_config(update_dict)

        # Set instance values
        self.filter_method = filter_method
        self.normalize_method = normalize_method
        self.transform_method = transform_method
        self.model_key = model_key
        self.classification_params = updated_params["classification_params"]
        self.model_params = self.classification_params[self.model_key]
        self.hyperparams = updated_params["current_params"]
        self.metadata_to_analyze = metadata_to_analyze
        self.global_outliers = global_outliers
        self.force_transform = force_transform
        self.force_extract = force_extract

        # Section to determine which operations can be skipped
        # Set params related to feature extraction and combination
        self.features = features_dict["features"] if features_dict is not None else None
        self.features_to_combine = features_dict["features_to_combine"] if features_dict is not None else None
        self.all_features = self._check_features()  # Flag to determine which features to handle

        # Initialize Transform and EEG-Extractor
        self._set_transforms_and_feature_extractor_instances(epoch_classes)

        # Get ResultIDs specified by the folder of initial_data_key
        self.patient_ids = self.pm.get_all_patient_ids(["initial_data", init_data_key])

        # Set run metadata class, collecting initial data
        filt_params_dict = {filter_method: updated_params["filtering_params"][filter_method]}
        transform_params_dict = {transform_method: updated_params["transform_params"][transform_method]}
        model_params_dict = {self.model_key: self.model_params}
        epoch_list = [self.class_0, self.class_1]  # Do not change this order. Index = label

        self.run_metadata_collector = RunMetadata(
            pm=self.pm,
            epoch_types=epoch_list,
            model_params=model_params_dict,
            initial_patient_ids=self.patient_ids,
            hyperparameters=self.get_current_hyperparams(),
            filtering_params=filt_params_dict,
            normalize_method=self.normalize_method,
            transform_params=transform_params_dict,
            classification_params=self.classification_params,
            run_name=run_name,
            force_overwrite=force_overwrite
        )

        # Set variables to use later
        self.split_paths = None
        self.metrics = None

    def complete_run(self, subworkflows_list: list[str] = None):
        """
        Executes the complete pipeline or a specified subset of workflow steps.

        :param subworkflows_list: List of keys (e.g. ["filter", "extract", "classify"]) to run only selected steps.
        """

        # Reference dict with all implemented steps. ORDER is important (order preservation in dict since Python 3.7)
        func_dict = {
            "filter": self.raw_eeg_filtering,
            "normalize": self.filtered_eeg_normalizing,
            "transform": self.transform_eeg_to_psd,
            "extract": self.feature_extraction,
            "combine": self.combine_features,
            "classify": self.split_classify_evaluate,
            "analyze": self.analyze_results
        }

        # Validation of the given list
        invalid = [step for step in subworkflows_list if step not in func_dict]
        if invalid:
            raise ValueError(f"Invalid subworkflow keys: {invalid}. Valid keys are: {list(func_dict)}")

        # If no subworkflow list given -> All steps will be carried out
        if subworkflows_list is None:
            selected_funcs = func_dict.items()
        else:  # Else create a custom list of steps, preserving the order of the reference dict
            selected_funcs = [
                (name, func)
                for name, func in func_dict.items()
                if name in subworkflows_list
            ]

        # Execute requested steps of the pipeline. All params of the steps are already defined in initialization
        for name, func in selected_funcs:
            print(f"Running step: {name}")
            func()

        # Collect remaining metadata and save them
        self._collect_remaining_pipeline_paramaters()
        self.run_metadata_collector.save_to_json()

    def already_calculated(self, calculation_type: str, epoch_type: str) -> bool:
        """
        Checks if the calculation type was already performed on given epoch type.
        :param calculation_type: The type of calculation. Allowed values are "extract_features" and "transform"
        :param epoch_type: Epoch type. Allowed values are "awake", "faw" and "normal_an".
        :return: True if calculation type was already performed on given epoch type, else False.
        """

        # load faw and awake data dependent of epochtypes
        parameters = self.get_current_hyperparams()

        if epoch_type == "normal_an":
            return False
        elif epoch_type == "awake":
            times_df = self.loader.load_awake_times_as_df(parameters)
        elif epoch_type == "faw":
            times_df = self.loader.load_faw_times_as_df(parameters)
        else:
            raise ValueError("Epoch type must be 'normal_an', 'awake' or 'faw'")

        if calculation_type == 'transform':
            psd_folderpath = self.pm.resolve_episode_path(parameters, epoch_type, ["features", "psds"], False, False)
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
                feature_filepath = self.pm.resolve_episode_path(
                    parameters, epoch_type, ["features", feature], True, False
                )
                comparison_dict = Comparison.compare_two_csv(feature_filepath, times_df)
                if not (comparison_dict["a_in_b"] and comparison_dict["b_in_a"]):
                    return False
            return True
        else:
            raise ValueError("Calculation type must be 'transform' or 'extract_features'")

    def raw_eeg_filtering(self):
        """ Applies filtering to all EEGs specified by the id-list in this class"""
        from MachineLearning.Preprocessing.filtering import Filtering

        filtering = Filtering(self.filter_method)
        filtering.filter_multiple_eeg(eeg_list=self.patient_ids)

    def filtered_eeg_normalizing(self):
        from MachineLearning.Preprocessing.normalizing import Normalizing
        normalizer = Normalizing(self.normalize_method)
        normalizer.normalize_multiple_eeg(eeg_list=self.patient_ids)

    def transform_eeg_to_psd(self):
        """Wrapper for transform function implemented in Transforms class"""
        if self.transformer is None:
            print("Skipping PSD transforms")
            return
        self.transformer.transform_eeg_episodes_to_psd()

    def feature_extraction(self):
        """
        Extracts defined features from all EEGs in the current patient_ids subset. Depending on the features
        given to this pipeline instance, it will extract features.
        """
        if self.feature_extractor is None:
            print("Skipping feature extraction")
            return

        feature_functions = self.feature_extractor.feature_extract_funcs

        # Calls all feature extraction functions
        if self.all_features:
            for function in feature_functions.values():
                function(self.feature_extractor)

        # Calls all functions implemented in the feature extractor
        else:
            for function_key in self.features:
                feature_functions[function_key](self.feature_extractor)

    def combine_features(self):
        """Wrapper for combining features method implemented in FeatureExtractor"""
        # No combination if no features given
        if self.features_to_combine is None:
            print("Skipping feature combination method")
            return

        # Ensure the feature extractor is not None
        if self.feature_extractor is None:
            epoch_tuple = self.class_0, self.class_1
            feature_extractor = EEGFeatureExtractor(self.pm, epoch_tuple, {})
        else:
            feature_extractor = self.feature_extractor

        # Combine features based on value in features_to_combine
        if self.features_to_combine == "all_features":
            feature_extractor.combine_features(True, [])
        elif not isinstance(self.features_to_combine, list):
            raise ValueError("Features must be either a list or a string with value 'all_features'")
        else:
            feature_extractor.combine_features(False, self.features_to_combine)

    def _set_transforms_and_feature_extractor_instances(self, epoch_classes: dict):
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

            # No parameters, since the pipeline already updated them and classes retrieve them from global config
            if not transform_epochs:
                self.transformer = None
            else:
                self.transformer = Transforms(tuple(transform_epochs), self.transform_method, None)
            if not feature_epochs:
                self.feature_extractor = None
            else:
                self.feature_extractor = EEGFeatureExtractor(self.pm, tuple(feature_epochs), None)

        # Ignore everything and initialize for all epochs if force operation is activated
        if self.force_transform:
            self.transformer = Transforms(tuple(epoch_classes.values()), self.transform_method, None)
        if self.force_extract:
            self.feature_extractor = EEGFeatureExtractor(self.pm, tuple(epoch_classes.values()), None)

    def _check_features(self) -> bool | None:
        """
        Checks self.features and returns None, True, or False based on the value of self.features.
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
            known_features = FeatureUtils.return_all_features(self.pm, "dict")
            # validate features given in list
            feature_keys = known_features.keys()
            for key in self.features:
                if key not in feature_keys:
                    raise ValueError(f"'{key}' is no valid feature key. Valid keys are: {feature_keys}")
            return False

    def _create_splits(
            self,
            test_size: float,
            random_state: int,
            split_paths=True,
            folds=True,
            iterations: int = None,
            remove_outlier_ids: bool = False,
            remove_epochs: bool = False,
            outlier_run_name: str = None,
            normalize_before_split: bool = True
    ):
        """
        Loads the test set, creates splits, splitting first on patient level and then tries to create equivalent
        ratios of faw and awake class in both test and train.

        :param test_size: Float that determines the ratio of test to train. E.g., 0.15 -> test:15% train:85%
        :param random_state: Randomness seed, to reproduce the shuffling of the splits.
        :param split_paths: If True, this method returns a tuple of paths leading to split train and test files.
        :param folds: If True, the splits will be as many non-overlapping folds as possible for cross-validation.
        :param iterations: Number of iterations for searching folds. Will be ignored if param "folds" is False.
        :param remove_outlier_ids: List of patient IDs to ignore when creating the splits.
        :param remove_epochs: If True, the splits will be created without the specified epochs.
        :param outlier_run_name: Name of the run to load the problematic IDs from.
        :return: If split_paths is True, returns the split paths: (<train set path>, <test set path>).
         Depending on folds, if it is true, a list of tuples will be returned, else a single tuple will be returned.
        """
        from MachineLearning.Evaluation.split_manager import SplitManager

        parameters = self.get_current_hyperparams()
        split_manager = SplitManager(parameters, self.class_0, self.class_1, test_size, random_state)
        split_manager.load_and_validate()
        if normalize_before_split:
            split_manager.normalize_data()

        if remove_outlier_ids:
            problematic_ids = self.loader.load_problematic_ids(parameters, self.model_key, outlier_run_name, self.global_outliers)
        else:
            problematic_ids = None

        if remove_epochs:
            problematic_epochs = self.loader.load_problematic_epochs(parameters, self.model_key, outlier_run_name, self.global_outliers)
        else:
            problematic_epochs = None

        # create single split or folds
        if folds:
            split_manager.create_custom_splits_by_test_size(
                min_iterations=iterations, ignore_ids=problematic_ids, ignore_epochs=problematic_epochs)
            return_splits = split_manager.return_k_fold_split_paths
        else:
            split_manager.create_single_split(ignore_ids=problematic_ids)
            return_splits = split_manager.return_split_paths

        if split_paths:
            return return_splits()
        return None

    def _run_svm_classifier(self, train_path: str, test_path: Path,
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

        parameters = self.get_current_hyperparams()
        svm_key = "svm"
        test_df = pd.read_csv(test_path)

        # Setup model
        if classifier is not None:
            clf = self.loader.load_model(svm_key, parameters)  # load pretrained model
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
            self.saver.save_model(clf, svm_key, parameters)

        # Save the original data with the prediction and error column
        if save_pred:
            self.saver.save_predicted_set(test_df, test_path, y_pred, parameters, svm_key)

        return y_pred, y_test, y_proba

    def _evaluate_metrics(self, y_test, y_pred, y_proba, folds: bool, print_metrics=True, save_metrics=True):
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
            prefix = "folds" if folds else "single"
            self.saver.save_ml_result(evaluation, "svm", self.get_current_hyperparams(), "dict", prefix, "metrics")

        return evaluation

    def split_classify_evaluate(self, folds=True):
        """
        Splits data, performs classification, and evaluates the results using a specified test size,
        random state, and optionally in a cross-validation setting.

        :param folds: Indicates whether to perform classification using cross-validation.
                      If True, it applies cross-validation; if False, a single split is used.
        :return: None
        """
        # Retrive classification params
        test_size = self.classification_params["test_size"]
        random_seed = self.classification_params["random_seed"]
        remove_outliers = self.classification_params["remove_outliers"]
        remove_outlier_epochs = self.classification_params["remove_outlier_epochs"]
        outlier_run_name = self.classification_params["outlier_run_name"] if (remove_outliers or remove_outlier_epochs) else None


        iterations = int(1 // test_size) * 2  # Double the number of minimal necessary iterations because I can :D
        self.split_paths = self._create_splits(
            test_size=test_size, random_state=random_seed, folds=folds, iterations=iterations,
            remove_outlier_ids=remove_outliers, remove_epochs=remove_outlier_epochs, outlier_run_name=outlier_run_name)

        if not folds:
            train_path, test_path = self.split_paths
            y_pred, y_test, y_proba = self._run_svm_classifier(train_path, test_path, save_clf=False, **self.model_params)
            self.metrics = self._evaluate_metrics(y_test, y_pred, y_proba, folds)

        else:
            y_pred, y_test, y_proba = self._collect_classification_results(self.split_paths, **self.model_params)
            self.metrics = self._evaluate_metrics(y_test, y_pred, y_proba, folds)

    @staticmethod
    def get_current_hyperparams() -> dict:
        """Returns the current parameters as a dictionary."""
        return load_config("parameters_config.yaml")["current_params"]

    @staticmethod
    def reset_hyperparams():
        """Resets the parameters stored in config i.e., globally."""
        default_params = load_config("parameters_config.yaml")["initial_params"]
        return update_config("parameters_config.yaml", {"current_params": default_params})

    def _collect_classification_results(self, split_paths: list, **kwargs) -> tuple[list, list, list]:
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
            prediction, true_label, probability = self._run_svm_classifier(
                train_path,
                test_path,
                save_clf=False,
                **kwargs
            )
            predictions.append(prediction)
            true_labels.append(true_label)
            probabilities.append(probability)

        return predictions, true_labels, probabilities

    def _analyze_single_result(self, result_path: Path, metadata_col: str, label: int, print_analysis=True, save_analysis=True,
                               plots=True):
        """
        Analyzes a single result file to evaluate the correlation between metadata and error rates, categorize errors by
        metadata groups, and assess the distribution of classes and confusion matrices for specified metadata.

        :param result_path: The file path to the CSV containing the results to be analyzed.
        :param metadata_col: The name of the metadata column in the results CSV to be analyzed.
        :param label: The label of the metadata column to be analyzed.
        :param print_analysis: A flag indicating whether the analysis results should be printed to the console.
        :param save_analysis: A flag indicating whether the analysis results should be saved to disk.
        :param plots: A flag indicating whether plots should be generated for the analysis results.
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
            filename = result_path.stem
            self.saver.save_metadata_analysis(error_correlation, "svm", self.get_current_hyperparams(),
                                         "dataframe", filename, "error_correlation")

            self.saver.save_metadata_analysis(error_by_metadata, "svm", self.get_current_hyperparams(),
                                         "dataframe", filename, f"error_by_{metadata_col}")

            self.saver.save_metadata_analysis(error_by_metadata, "svm", self.get_current_hyperparams(),
                                         "dataframe", filename, f"error_label_{label}_by_{metadata_col}")

            self.saver.save_metadata_analysis(class_dist_per_metadata, "svm", self.get_current_hyperparams(),
                                         "dataframe", filename, f"class_dist_per_{metadata_col}")

            self.saver.save_metadata_analysis(confusion_matrices_by_metadata, "svm", self.get_current_hyperparams(),
                                         "dict", filename, f"confusion_matrices_by_{metadata_col}")

            if plots:
                self.saver.save_metadata_analysis(error_dist_by_metadata, "svm", self.get_current_hyperparams(),
                                             "plot", filename, f"error_dist_by_{metadata_col}")

                self.saver.save_metadata_analysis(temp_error_by_metadata, "svm", self.get_current_hyperparams(),
                                             "plot", filename, f"temp_error_by_{metadata_col}")

        print("Analysis complete.")

    def _analyze_meta_analyses(self, model_key: str, metadata_col, print_analysis=True, save_analysis=True, plots=True):
        """
        Analyzes the metadata results from multi-fold analysis and generates
        aggregated errors, balances, and plots. Optionally, the results can be
        printed or saved for further examination.

        :param model_key: The key identifier for the model being analyzed.
        :type model_key: str
        :param metadata_col: Column from metadata used for aggregation or plotting.
        :type metadata_col: Any
        :param print_analysis: Flag indicating whether to print analysis results, defaults to True.
        :type print_analysis: bool, optional
        :param save_analysis: Flag indicating whether to save the analysis results, defaults to True.
        :type save_analysis: bool, optional
        :param plots: Flag indicating whether to generate and save plots, defaults to True.
        :type plots: bool, optional
        :return: None
        """

        from MachineLearning.Evaluation.meta_fold_analyzer import MetaFoldAnalyzer

        print("Analyzing single fold analysis results")

        # Initialize analyzer
        parameters = self.get_current_hyperparams()
        fold_analyzer = MetaFoldAnalyzer(self.pm, model_key, parameters)
        fold_analyzer.load_all_folds(metadata_col)
        # Carry out analysis
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
            self.saver.save_metadata_analysis(agg_err_by_group, "svm", parameters, "dataframe", "Summary_analysis",
                                         "agg_error_by_groups")
            self.saver.save_metadata_analysis(agg_label_err_by_group, "svm", parameters, "dataframe", "Summary_analysis",
                                         "agg_label_error_by_groups")
            self.saver.save_metadata_analysis(acc_vs_class_dist, "svm", parameters, "dataframe", "Summary_analysis",
                                         "acc_vs_class_distribution")
            self.saver.save_metadata_analysis(prec_vs_class_dist, "svm", parameters, "dataframe", "Summary_analysis",
                                         "prec_vs_class_distribution")
            self.saver.save_metadata_analysis(rec_vs_class_dist, "svm", parameters, "dataframe", "Summary_analysis",
                                         "rec_vs_class_distribution")
            if plots:
                self.saver.save_metadata_analysis(error_heatmap, "svm", parameters, "plot", "Summary_analysis",
                                             "error_heatmap")

        print("Analysis of single analysis results complete")

    def analyze_results(self, print_analysis=True, save_analysis=True, plots=False):

        # Gather all results
        parameter_dict = self.get_current_hyperparams()
        result_folder = self.pm.get_complex_ml_path(parameter_dict, ["results", self.model_key], False, False)
        path_list, _ = PathUtils.list_files_in_folder(result_folder, ".csv", fullpaths=True)

        # Analyze all given metadata in all results
        for metadata in self.metadata_to_analyze:
            for result_path in path_list:
                self._analyze_single_result(result_path, metadata, 1, print_analysis, save_analysis, plots)

            # Summary Analysis of single analysis results
            self._analyze_meta_analyses(self.model_key, metadata, print_analysis, save_analysis, plots)

    def _collect_remaining_pipeline_paramaters(self):
        self.run_metadata_collector.set_feature_info()
        self.run_metadata_collector.set_split_data(self.split_paths)
        self.run_metadata_collector.set_metrics(self.metrics)


    @staticmethod
    def _update_param_config(update_params: dict) -> dict:
        """Updates the parameters stored in config i.e., globally and returns the updated parameters as a dictionary."""
        return update_config("parameters_config.yaml", update_params)