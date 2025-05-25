from MachineLearning.Features.eeg_feature_extractor import EEGFeatureExtractor


def runner():
    """
    Function to execute any code
    """

    # Create an instance of EEGFeatureExtractor
    extractor = EEGFeatureExtractor()

    # Extract PSD features
    # print("Starting PSD extraction...")
    # extractor.extract_psd(channel=1, nperseg_seconds=2)

    # extractor.extract_relative_bandpower_for_parameter_combination()
    # extractor.extract_shannon_entropy_for_parameter_combination()
    # extractor.extraxt_spectral_skewness_for_parameter_combination()
    extractor.extraxt_spectral_kurtosis_for_parameter_combination()


if __name__ == "__main__":
    runner()
