"""
Here is the place to test any code.
"""
from MachineLearning.Utils.config_handler import load_config


if __name__ == '__main__':
    loaded_config = load_config("parameters_config.yaml")
    print(loaded_config)
