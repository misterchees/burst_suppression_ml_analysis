from MachineLearning.Utils.feature_utils import FeatureUtils


class Epochs:

    def __init__(self, parameters: dict = None, channel: int = None, filtered=True, faw=True):
        self.epoch_times = []  # Epochs of current parameters
        self.parameters = parameters  # Parameters of epochs containing metadata
        if parameters:
            self.epoch_times = FeatureUtils.return_eeg_epochs(parameters, channel=channel, faw=faw)
        self.channel = channel  # Determines the channel of the EEG, from where the epochs are
        self.filtered = filtered  # Determines if epochs are from filtered or raw EEG
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

    def update_epochs(self, parameters: dict = None, channel: int = None, filtered=True, faw=True):
        if parameters:
            self.epoch_times = FeatureUtils.return_eeg_epochs(parameters, channel=channel, faw=faw)
        self.parameters = parameters
        self.channel = channel
        self.filtered = filtered
        self.faw = faw
