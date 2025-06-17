from MachineLearning.Utils.feature_utils import FeatureUtils


class Epochs:

    def __init__(self, parameters: dict = None, channel: int = None, filtered=True, faw=True,
                 normal_an=False, num_an: int = None):
        self.epoch_times = []  # Epochs of current parameters
        self.parameters = parameters  # Parameters of epochs containing metadata
        if parameters:
            self.epoch_times = FeatureUtils.return_eeg_epochs(parameters, channel=channel, faw=faw,
                                                              normal_an=normal_an, num_an=num_an)
        self.channel = channel  # Determines the channel of the EEG, from where the epochs are
        self.filtered = filtered  # Determines if epochs are from filtered or raw EEG
        self.faw = faw  # Determines if epochs from fake awakeness or true awakeness
        self.normal_an = normal_an
        self.num_an = num_an

    def is_empty(self) -> bool:
        return not self.epoch_times

    def update_if_necessary(self, parameters: dict = None, channel: int = None, filtered: bool = True, faw: bool = True,
                            normal_an: bool = False, num_an: int = None):
        """
        Checks if epochs are empty and if any of the other parameters differ from the object parameters. If one of these
        conditions is fullfilled, it will update itself be loading the epochs for given parameters.
        :param parameters: Parameters that define which epochs to choose
        :param channel: EEG channel
        :param filtered: If True loads from filtered EEG, else from raw EEG.
        :param faw: If True loads from faw EEG, else from raw EEG.
        :param normal_an: If True samples from normal anesthesia EEG.
        :param num_an: Number of samples i.e. number of Epochs.
        """
        if normal_an:
            if self.is_empty() or num_an != self.num_an:
                self.update_epochs(parameters, channel=channel, filtered=filtered, normal_an=normal_an, num_an=num_an)
        else:
            if (self.is_empty() or parameters != self.parameters or channel != self.channel
                    or filtered != self.filtered or faw != self.faw):
                self.update_epochs(parameters, channel, filtered, faw)

    def update_epochs(self, parameters: dict = None, channel: int = None, filtered=True, faw=True,
                      normal_an: bool = False, num_an: int = None):
        if parameters:
            self.epoch_times = FeatureUtils.return_eeg_epochs(parameters, channel=channel, faw=faw,
                                                              normal_an=normal_an, num_an=num_an)
        self.parameters = parameters
        self.channel = channel
        self.filtered = filtered
        self.faw = faw
        self.normal_an = normal_an
        self.num_an = num_an
