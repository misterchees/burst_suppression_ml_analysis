from MachineLearning.Utils.config_loader import load_config

config = load_config("path_config.yaml")

print(config["base_dir"]["subdirs"].keys())
