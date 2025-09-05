"""
Here is the place to test any code.
"""
from MachineLearning.Core.runner import generate_feature_combinations

runs = generate_feature_combinations()
for triple in runs:
    print(triple)
    print("###############################")
