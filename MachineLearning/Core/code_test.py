"""
Here is the place to test any code.
"""

import pandas as pd
from MachineLearning.Models.svm_classifier import SVMClassifier
from MachineLearning.Evaluation.metrics_evaluator import MetricsEvaluator
from MachineLearning.Evaluation.split_manager import SplitManager
from MachineLearning.Core.ml_object import MLObject


# Load splits
ml_object = MLObject(True, True, True)
split_manager = SplitManager(ml_object.parameter_dict)
train_path, test_path = split_manager.return_split_paths()
print(train_path)
print(test_path)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Prepare features/labels
X_train = train_df.drop(columns=["Start", "End", "ResultID", "label"]).values
y_train = train_df["label"].values

X_test = test_df.drop(columns=["Start", "End", "ResultID", "label"]).values
y_test = test_df["label"].values

# Train model
clf = SVMClassifier(probability=True)
clf.train(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Evaluate
evaluator = MetricsEvaluator(y_test, y_pred, y_proba)
results = evaluator.evaluate()

print(results)


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