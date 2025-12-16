"""
Here is the place to test any code.
"""
from MachineLearning.IO.load_data import LoadData
from MachineLearning.Preprocessing.filtering import Filtering
import pandas as pd

loader = LoadData()
filter_instance = Filtering("butterworth")
test_params = {"fixed_window_size": 20}


def read_test():
    filter_instance.butterworth(74, 0.5, 30, 4)



if __name__ == "__main__":
    read_test()