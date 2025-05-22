class MLObject:
    merged_episodes = False  # flag to determine if episodes are merged
    bis_threshold = 70  # lower threshold on BIS value (options: 70)
    mac_threshold = 0.8  # lower threshold on MAC value (options: 0.5, 0.6, 0.7, 0.8)
    min_episode_length = 20  # lower threshold on episode length (options: 5, 6, 7, 8, 9, 10, 15, 20)
    refractory_time = 5  # maximum refractory time between episodes in seconds (options: 3, 4, 5)
    fixed_window_size = 20  # exact window length (options: 5, 6, 7, 8, 9, 10, 15, 20)
    overlap = 0.0  # window overlap (options: 0.0, 0.25, 0.5)

    eeg_fs = "fs"
    eeg_rawEEG = "rawEEG"

    psd_freq_col = "Frequency_Hz"
    psd_power_col = "PSD_V2_per_Hz"

    # Typical bands of EEG
    frequency_bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 45)
    }

    def __init__(self, **kwargs):
        """
        Initialize the EEGFeatureExtractor with optional variables. It checks for every attribute in
        EEGFeatureExtractor and uses the initialized default if no value is given in the kwargs.
        """
        for attr in ["preprocessing_dir", "vitaldb_eeg_dir", "output_dir", "merged_episodes", "bis_threshold",
                     "mac_threshold", "min_episode_length", "refractory_time", "fixed_window_size", "overlap"]:
            setattr(self, attr, kwargs.get(attr, getattr(self.__class__, attr)))
