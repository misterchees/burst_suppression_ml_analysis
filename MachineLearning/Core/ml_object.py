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

    param_config = load_config("parameters_config.yaml")
    parameter_dict = param_config["initial_params"]

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
        Sets any number of the attributes in the EEGFeatureExtractor.

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
        if "faw" in self.epoch_types:
            self.faw_epochs.update_if_necessary(self.parameter_dict, channel=channel)
        if "awake" in self.epoch_types:
            self.awake_epochs.update_if_necessary(self.parameter_dict, channel=channel)
        if "normal_an" in self.epoch_types:
            awake_epochs = self.awake_epochs.epoch_times
            faw_epochs = self.faw_epochs.epoch_times

            if not awake_epochs and not faw_epochs:
                raise Exception("To compare with anesthesia, episodes from another type must be present "
                                "to align the number of episodes.")
            number_of_epochs = max(len(awake_epochs), len(faw_epochs))
            self.normal_an_epochs.update_if_necessary(self.parameter_dict, num_an=number_of_epochs, channel=channel)


