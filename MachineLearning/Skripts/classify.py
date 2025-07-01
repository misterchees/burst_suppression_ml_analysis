"""This script is used to train a SVM classifier on the training set and evaluate it on the test set."""
import pandas as pd

from MachineLearning.Models.svm_classifier import SVMClassifier
from MachineLearning.IO.load_data import LoadData
from MachineLearning.Utils.config_loader import load_config

parameter_dict = load_config("parameters_config.yaml")["initial_params"]
classifier = SVMClassifier()
loader = LoadData()

train_path = loader.return_single_split_folder_fullpath(parameter_dict, "train", False)
test_path = loader.return_single_split_folder_fullpath(parameter_dict, "test", False)

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