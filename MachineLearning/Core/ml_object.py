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
        Create an instance with new values for the parameters

        :param kwargs: Any parameter with value
        """
        # copy of parameters for these instance
        self.parameter = self.parameter_dict.copy()

        for key in kwargs:
            if key in self.parameter:
                self.parameter[key] = kwargs[key]
