"""
Module containing MlObject class.
"""
from MachineLearning.Utils.config_loader import load_config
from MachineLearning.Utils.epochs import Epochs


class MLObject:
    """
    Superclass for all machine learning objects. Especially Feature Extractor and Transformer.
    Manages Epochs that are analyzed.
    """
    parameter_dict = {
        "merged_episodes": False,  # flag to determine if episodes are merged
        "bis_threshold": 70,  # lower threshold on BIS value (options: 70)
        "mac_threshold": 0.8,  # lower threshold on MAC value (options: 0.5, 0.6, 0.7, 0.8)
        "min_episode_length": 20,  # lower threshold on episode length (options: 5, 6, 7, 8, 9, 10, 15, 20)
        "refractory_time": 5,  # maximum refractory time between episodes in seconds (options: 3, 4, 5)
        "fixed_window_size": 20,  # exact window length (options: 5, 6, 7, 8, 9, 10, 15, 20)
        "overlap": 0.0  # window overlap (options: 0.0, 0.25, 0.5)
    }

    param_config = load_config("parameters_config.yaml")

    VALID_EPOCH_TYPES = ["faw", "awake", "normal_an"]
    faw_epochs = Epochs("faw")
    awake_epochs = Epochs("awake")
    normal_an_epochs = Epochs("normal_an")

    def __init__(self, *epoch_types, **parameter_kwargs):
        """
        Create an instance with global values for Epochs, flags and parameters.
        :param epoch_types: Defines which Epochs will be handled by this instance. Valid options are:
        "faw", "awake", "normal_an"
        :param parameter_kwargs: Any keyword parameter(s)
        """
        for element in epoch_types:
            if element not in self.VALID_EPOCH_TYPES:
                raise ValueError(f"Invalid epoch type: {element}. Valid epoch types are {self.VALID_EPOCH_TYPES}")
        self.epoch_types = epoch_types

        for key in parameter_kwargs:
            if key in self.parameter_dict:
                self.parameter_dict[key] = parameter_kwargs[key]
        print(f"Parameters: {self.parameter_dict}. Epoch_types: {self.epoch_types}")

    def set_attributes(self, *epoch_types, **parameter_kwargs):
        """
        Sets any number of the attributes of the EEGFeatureExtractor.

        :param epoch_types: Defines which Epochs will be handled by this instance. Valid options are:
        "faw", "awake", "normal_an"
        :param parameter_kwargs: Any keyword parameter(s)
        """
        for element in epoch_types:
            if element not in self.VALID_EPOCH_TYPES:
                raise ValueError(f"Invalid epoch type. Valid epoch types are {self.VALID_EPOCH_TYPES}")
        self.epoch_types = epoch_types

        for key in parameter_kwargs:
            if key in self.parameter_dict:
                print(f"Changing {key} from {self.parameter_dict[key]} to {parameter_kwargs[key]} ")
                self.parameter_dict[key] = parameter_kwargs[key]

    def update_current_epochs(self, channel: int):
        """
        Calls the update function of all epoch objects.
        :param channel: current channel of the EEG from where the epochs are.
        """
        # Get all Episodes for all handled epochs
        for epoch in self.epoch_types:
            if epoch == "normal_an":
                if len(self.epoch_types) <= 1:
                    raise Exception("To compare with anesthesia, episodes from another type must be present "
                                    "to align the number of episodes.")
                number_of_epochs = max(len(self.awake_epochs.epoch_times), len(self.faw_epochs.epoch_times))
                self.normal_an_epochs.update_if_necessary(self.parameter_dict, num_an=number_of_epochs, channel=channel)
            elif epoch == "faw":
                self.faw_epochs.update_if_necessary(self.parameter_dict, channel=channel)
            else:
                self.awake_epochs.update_if_necessary(self.parameter_dict, channel=channel)
