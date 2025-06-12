from MachineLearning.Utils.feature_utils import FeatureUtils


class Epochs:
    feature_utils = FeatureUtils()
    epoch_times = []  # List of epochs

    def __init__(self, parameters: dict = None, channel: int = None, filtered: bool = True, faw: bool = True):
        if parameters:
            self.epoch_times = self.feature_utils.return_eeg_epochs(parameters, channel=channel)
            self.parameters = parameters  # Parameters of epochs containing metadata
        if channel:
            self.channel = channel  # Determines the channel of the EEG, from where the epochs are
        if filtered:
            self.filtered = filtered  # Determines if epochs are from filtered or raw EEG
        if faw:
            self.faw = faw  # Determines if epochs from fake awakeness or true awakeness

    def is_empty(self) -> bool:
        return not self.epoch_times

    def update_if_necessary(self, parameters: dict = None, channel: int = None, filtered: bool = True, faw: bool = True):
        """
        Checks if epochs are empty and if any of the other parameters differ from the object parameters. If one of these
        conditions is fullfilled, it will update itself be loading the epochs for given parameters.
        :param parameters: Parameters that define which epochs to choose
        :param channel: EEG channel
        :param filtered: If True loads from filtered EEG, else from raw EEG.
        :param faw: If True loads from faw EEG, else from raw EEG.
        """
        if (self.is_empty() or parameters != self.parameters or channel != self.channel
                or filtered != self.filtered or faw != self.faw):
            self.update_epochs(parameters, channel, filtered, faw)

    def update_epochs(self, parameters: dict = None, channel: int = None, filtered: bool = True, faw: bool = True):
        if parameters:
            self.epoch_times = self.feature_utils.return_eeg_epochs(parameters, channel=channel)
            self.parameters = parameters
        if channel:
            self.channel = channel
        if filtered:
            self.filtered = filtered
        if faw:
            self.faw = faw
