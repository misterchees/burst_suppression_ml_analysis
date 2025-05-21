from MachineLearning.FeatureExtraction.eeg_feature_extractor import EEGFeatureExtractor


def main():
    """
    Main entry point for EEG Machine Learning Workflow
    """

    # Create an instance of EEGFeatureExtractor
    extractor = EEGFeatureExtractor()

    # Extract PSD features
    # print("Starting PSD extraction...")
    # extractor.extract_psd(channel=1, nperseg_seconds=2)

    extractor.extract_relative_bandpower()
    # extractor.extract_shannon_entropy()


if __name__ == "__main__":
    main()
