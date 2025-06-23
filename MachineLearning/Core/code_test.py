"""
Here is the place to test any code.
"""

from MachineLearning.Evaluation.split_manager import SplitManager
from MachineLearning.Models.svm_classifier import SVMClassifier
from MachineLearning.Models.ParamTuning.svm_grid_search import SVMGridSearch

parameter_dict = {
    "merged_episodes": False,  # flag to determine if episodes are merged
    "bis_threshold": 70,  # lower threshold on BIS value (options: 70)
    "mac_threshold": 0.8,  # lower threshold on MAC value (options: 0.5, 0.6, 0.7, 0.8)
    "min_episode_length": 20,  # lower threshold on episode length (options: 5, 6, 7, 8, 9, 10, 15, 20)
    "refractory_time": 5,  # maximum refractory time between episodes in seconds (options: 3, 4, 5)
    "fixed_window_size": 20,  # exact window length (options: 5, 6, 7, 8, 9, 10, 15, 20)
    "overlap": 0.0  # window overlap (options: 0.0, 0.25, 0.5)
}

split_manager = SplitManager(parameter_dict, "faw", "awake", test_size=0.15)
split_manager.load_and_validate()
X, y, splits = split_manager.create_cv_splits()
svm = SVMClassifier()
svm_base_model = svm.get_base_model()
metric_list = ["accuracy", "recall", "precision"]
for metric in metric_list:
    grid_search = SVMGridSearch(svm_base_model, X, y, splits, metric)
    print(f"Results for grid search to optimize {metric}:")
    print(f"Best estimator: {grid_search.best_estimator()}")
    print(f"Best score: {grid_search.best_score()}")
    print(f"Best params: {grid_search.best_params()}")

"""
from MachineLearning.IO.io_core import IOCore
from MachineLearning.Utils.path_utils import PathUtils
from MachineLearning.Core.ml_object import MLObject
io_core = IOCore()
ml_object = MLObject(True, True)
parameters = MLObject.parameter_dict

psd_dir = io_core.return_folder_path("features", "psds")
abcd_subdir = PathUtils.return_A_B_C_D_name("PSD", parameters)
xy_subdir = PathUtils.return_X_Y_name(parameters)
psd_dir_fullpath = PathUtils.return_anypath(psd_dir, abcd_subdir, xy_subdir)

print(io_core.return_all_parameter_fullpath(parameters, False, True, "features", "psds"))
print(psd_dir_fullpath)
"""
