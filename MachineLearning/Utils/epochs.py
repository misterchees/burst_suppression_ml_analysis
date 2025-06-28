"""this module contains the Epoch Class"""
from MachineLearning.Utils.feature_utils import FeatureUtils


class Epochs:
    """
    Handles all epochs of a single epoch type and given parameters. Type can't be changed after initialization
    The most important attribute here is epoch_times. This list contains all epochs with metadata as a tuple.
    The tuple is of the following structure:
    start_time, end_time, result_id, fs, eeg_segment
    """

    def __init__(self, epoch_type: str,  parameters: dict = None, channel: int = None, filtered=True,
                 num_an: int = None):
        """
        Initializes a new Epochs object that contains the read EEG snippets based on the times of the epoch.
        Its main purpose is to load the epochs only once to perform all necessary manipulations on them
        without the need to load them multiple times. Can be updated if parameters change.

        :param epoch_type: Type of the stored epochs. Valid options are 'faw','awake' and 'normal_an'.
        :param parameters: Dictionary of parameters, that determine the epochs.
        :param channel: Channel of the EEG from which the epochs are read. Valid options are 1 or 2.
        :param filtered: If true, epochs are from a filtered EEG, else from the raw EEG.
        :param num_an: Will be ignored if epoch_type is not 'normal_an'. Number of normal anesthesia epochs,
        since these are randomly sampled. It should be at least the same number of the epochs,
        which will be compared to these.
        """
        self.epoch_times = []  # Epochs of current parameters
        self.parameters = parameters  # Parameters of epochs containing metadata
        if parameters:
            self.epoch_times = FeatureUtils.return_eeg_epochs(epoch_type, parameters, channel=channel, num_an=num_an)
        self.channel = channel  # Determines the channel of the EEG, from where the epochs are
        self.filtered = filtered  # Determines if epochs are from filtered or raw EEG
        self.epoch_type = epoch_type
        self.num_an = num_an

    def is_empty(self) -> bool:
        """Makes use of the behavior that boolean checks return false for an empty list."""
        return not self.epoch_times

    def update_if_necessary(self, parameters: dict = None, channel: int = None, filtered: bool = True, num_an: int = None):
        """
        Checks if epochs are empty and if any of the other parameters differ from the current object parameters.
        If one of these conditions is fullfilled, it will update itself be loading the epochs for given parameters.
        :param parameters: Parameters to check against current parameters.
        :param channel: EEG channel to check against current channel (Valid options: 1 or 2).
        :param filtered: If True loads from filtered EEG, else from raw EEG.
        :param num_an: Number of normal anesthesia epochs to check against the current number.
        """
        if self.epoch_type == "normal_an":
            if self.is_empty() or num_an != self.num_an:
                self.update_epochs(parameters, channel=channel, filtered=filtered, num_an=num_an)
        else:
            if (self.is_empty() or parameters != self.parameters or channel != self.channel
                    or filtered != self.filtered):
                self.update_epochs(parameters, channel, filtered)

    def update_epochs(self, parameters: dict = None, channel: int = None, filtered=True, num_an: int = None):
        """
        Updates this object by loading epochs of given parameters. If no parameters were
        :param parameters: Parameters that define which epochs to choose.
        :param channel: Channel of the EEG from which the epochs are loaded.
        :param filtered: If True loads from filtered EEG, else from raw EEG
        :param num_an: Will be ignored if this instance does not handle normal anesthesia epochs.
        Number of epochs of normal anesthesia to be sampled.
        """
        if parameters:
            self.epoch_times = FeatureUtils.return_eeg_epochs(self.epoch_type, parameters, channel=channel, num_an=num_an)
        else:
            raise ValueError("No parameters given, no epochs will be updated!")
        self.parameters = parameters
        self.channel = channel
        self.filtered = filtered
        self.num_an = num_an
