"""
Here is the place to test any code.
"""
from MachineLearning.Utils.config_loader import load_config
Parameter_dict = load_config("parameters_config.yaml")["initial_params"]

"""
from MachineLearning.Evaluation.split_manager import SplitManager
from MachineLearning.Models.svm_classifier import SVMClassifier
from MachineLearning.Models.ParamTuning.svm_grid_search import SVMGridSearch

split_manager = SplitManager(parameter_dict, "faw", "awake", test_size=0.15)
split_manager.load_and_validate()
X, y, splits = split_manager.create_custom_splits_by_test_size()
svm = SVMClassifier()
svm_base_model = svm.get_base_model()
metric_list = ["accuracy", "recall", "precision"]
for metric in metric_list:
    grid_search = SVMGridSearch(svm_base_model, X, y, splits, metric)
    grid_search.run()
    print(f"Results for grid search to optimize {metric}:")
    print(f"Best estimator: {grid_search.best_estimator()}")
    print(f"Best score: {grid_search.best_score()}")
    print(f"Best params: {grid_search.best_params()}")
"""