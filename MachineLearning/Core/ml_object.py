from MachineLearning.Utils.config_loader import load_config
from MachineLearning.Utils.epochs import Epochs


class MLObject:
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

    faw: bool
    awake: bool
    faw_epochs = Epochs()
    awake_epochs = Epochs()

    def __init__(self, faw: bool, awake: bool, **kwargs):
        """
        Create an instance with new values for the parameters

        :param kwargs: Any keyword parameter(s)
        :param faw: Boolean value to indicate if instance currently handles fake awake data.
        :param awake: Boolean value to indicate if instance currently handles awake data.
        """
        self.faw = faw
        self.awake = awake

        # copy of parameters for this instance
        self.parameter = self.parameter_dict.copy()

        for key in kwargs:
            if key in self.parameter:
                self.parameter[key] = kwargs[key]
        print(f"Parameters: {self.parameter}. Flags: faw={self.faw}, awake={self.awake}")

    def set_attributes(self, faw: bool = None, awake: bool = None, **kwargs):
        """
        Sets any number of the attributes of the EEGFeatureExtractor.

        :param kwargs: Any keyword parameter(s)
        :param faw: Boolean value to indicate if instance currently handles fake awake (True) or awake (False) data.
        :param awake: Boolean value to indicate if instance currently handles awake data.
        """
        if faw:
            self.faw = faw
        if awake:
            self.awake = awake

        # copy of parameters for these instance
        self.parameter = self.parameter_dict.copy()

        for key in kwargs:
            if key in self.parameter:
                self.parameter[key] = kwargs[key]

    def update_current_epochs(self, channel: int):
        # Get all Episodes for awake and fake awake epochs if necessary (default = filtered episodes)
        if self.faw:
            self.faw_epochs.update_if_necessary(self.parameter_dict, channel)
        if self.awake:
            self.awake_epochs.update_if_necessary(self.parameter_dict, channel, faw=False)
