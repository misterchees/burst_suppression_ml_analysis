from MachineLearning.Utils.feature_utils import FeatureUtils


class Epochs:
    feature_utils = FeatureUtils()
    epoch_times = []

    def __init__(self, parameters: dict = None, channel: int = None, filtered: bool = True):
        if parameters:
            self.epoch_times = self.feature_utils.return_faw_eeg_epochs(parameters, channel=channel)
            self.parameters = parameters
        if channel:
            self.channel = channel
        if filtered:
            self.filtered = filtered

    def is_empty(self) -> bool:
        return not self.epoch_times

    def update_if_necessary(self, parameters: dict = None, channel: int = None, filtered: bool = True):
        """
        Checks if epochs are empty and if any of the other parameters differ from the object parameters. If one of this
        conditions is fullfilled, it will update itself be loading the epochs for given parameters.
        :param parameters: Parameters that define which epochs to choose
        :param channel: EEG channel
        :param filtered: If True loads from filtered EEG, else from raw EEG.
        """
        if self.is_empty() or parameters != self.parameters or channel != self.channel or filtered != self.filtered:
            self.update_epochs(parameters, channel, filtered)

    def update_epochs(self, parameters: dict = None, channel: int = None, filtered: bool = True):
        if parameters:
            self.epoch_times = self.feature_utils.return_faw_eeg_epochs(parameters, channel=channel)
            self.parameters = parameters
        if channel:
            self.channel = channel
        if filtered:
            self.filtered = filtered
