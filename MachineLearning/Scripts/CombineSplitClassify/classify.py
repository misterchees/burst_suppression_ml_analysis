"""This script is used to train a SVM classifier on the training set and evaluate it on the test set."""
import pandas as pd

from MachineLearning.Models.svm_classifier import SVMClassifier
from MachineLearning.IO.load_data import LoadData
from MachineLearning.Utils.config_handler import load_config
from MachineLearning.Utils.path_manager import PathManager

parameter_dict = load_config("parameters_config.yaml")["initial_params"]
classifier = SVMClassifier()
loader = LoadData()
pm = PathManager()

split_dir = pm.get_complex_ml_path(
    parameter_dict, ["test_and_train_data", "splits"], False, False
)
train_path = split_dir / "train_split.csv"
test_path = split_dir / "test_split.csv"

# Prepare features/labels
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["Start", "End", "ResultID", "label"]).values
y_train = train_df["label"].values

# Train model
clf = SVMClassifier(probability=True)
clf.train(X_train, y_train)

# Drop metadata and leave the label as the only column in the test set
X_test = test_df.drop(columns=["Start", "End", "ResultID", "label"]).values
y_test = test_df["label"].values

# Predict
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

print("Successful")