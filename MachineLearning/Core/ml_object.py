"""
Module containing MlObject class.
"""
from MachineLearning.IO.load_data import LoadData
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Utils.config_handler import load_config, update_config
from MachineLearning.Utils.epochs import Epochs
from MachineLearning.Utils.path_manager import PathManager


class MLObject:
    """
    Superclass for all machine learning objects. Especially Feature Extractor and Transformer.
    Manages Epochs that are analyzed.
    """

    PARAM_CONFIG_FILENAME = "parameters_config.yaml"
    param_config = load_config(PARAM_CONFIG_FILENAME)

    VALID_EPOCH_TYPES = ["faw", "awake", "normal_an"]
    faw_epochs = Epochs("faw")
    awake_epochs = Epochs("awake")
    normal_an_epochs = Epochs("normal_an")

    def __init__(self,pm: PathManager, epoch_types: tuple, parameter_update: dict):
        """
        Create an instance with global values for Epochs, flags and parameters.
        :param epoch_types: Defines which Epochs will be handled by this instance. Valid options are:
        "faw", "awake", "normal_an"
        :param parameter_update: A dict containing the parameters to be updated.
        """

        # Initialize IO handlers
        self.pm = pm
        self.loader = LoadData(self.pm)
        self.saver = SaveResult(self.pm)


        for element in epoch_types:
            if element not in self.VALID_EPOCH_TYPES:
                raise ValueError(f"Invalid epoch type: {element}. Valid epoch types are {self.VALID_EPOCH_TYPES}")
        self.epoch_types = epoch_types

        if parameter_update is not None:
            self.parameter_dict = update_config(self.PARAM_CONFIG_FILENAME, parameter_update)["current_params"]
        else:
            self.parameter_dict = self.param_config["current_params"]
        print(f"Parameters: {self.parameter_dict}. Epoch_types: {self.epoch_types}")

    def set_attributes(self, epoch_types: tuple, parameter_update: dict):
        """
        Sets any number of the attributes in the EEGFeatureExtractor.

        :param epoch_types: Defines which Epochs will be handled by this instance. Valid options are:
        "faw", "awake", "normal_an"
        :param parameter_update: A dict containing the parameters to be updated.
        """
        for element in epoch_types:
            if element not in self.VALID_EPOCH_TYPES:
                raise ValueError(f"Invalid epoch type. Valid epoch types are {self.VALID_EPOCH_TYPES}")
        self.epoch_types = epoch_types

        self.parameter_dict = update_config(self.PARAM_CONFIG_FILENAME, parameter_update)["current_params"]
        print(f"Parameters: {self.parameter_dict}. Epoch_types: {self.epoch_types}")

    def update_current_epochs(self, channel: int):
        """
        Calls the update function of all epoch objects.
        :param channel: Current channel of the EEG from where the epochs are.
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


