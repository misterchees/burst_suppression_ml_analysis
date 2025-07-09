"""
Here is the place to test any code.
"""
from MachineLearning.Utils.config_handler import load_config

parameter_dict = load_config("parameters_config.yaml")["initial_params"]
path_to_file = load_config("path_config.yaml")["base_dir"]["path_name"]

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

from MachineLearning.Evaluation.metrics_evaluator import MetricsEvaluator
from MachineLearning.IO.load_data import LoadData
from MachineLearning.Utils.config_handler import update_config

loader = LoadData()
evaluator = MetricsEvaluator(None, None, None, None, None)

overlaps = [0.0, 0.25, 0.5]
min_episode_lengths = [10, 15]

for overlap in overlaps:
    for ep_length in min_episode_lengths:
        new_params = {
            "current_params": {
                "overlap": overlap,
                "min_episode_length": ep_length,
                "fixed_window_size": ep_length
            }
        }
        curent_params = update_config("parameters_config.yaml", new_params)["current_params"]
        print(f"\n#######Testing Parameters: {curent_params}\n")
        current_metrics = loader.load_metrics(curent_params, "svm")
        evaluator.print_result(current_metrics["summary"], True)

new_params = {
            "current_params": {
                "overlap": 0,
                "min_episode_length": 20,
                "fixed_window_size": 20
            }
}
curent_params = update_config("parameters_config.yaml", new_params)["current_params"]
print(f"\n#######Testing Parameters: {curent_params}\n")
current_metrics = loader.load_metrics(curent_params, "svm")
evaluator.print_result(current_metrics["summary"], True)
