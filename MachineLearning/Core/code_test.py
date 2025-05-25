from MachineLearning.Features.eeg_feature_extractor import EEGFeatureExtractor
from MachineLearning.Preprocessing.filtering import Filtering
import pandas as pd

random_PSD_path = "C:\\Users\\jesus\\OneDrive\\Dokumente\\Jesús\\Studium\\Fächer - Bioinformatik\\Praktische Arbeit und Bachelorarbeit\\Material\\Daten\\Features\\PSDs\\PSD_70_080_20_5\\Summary_Episodes_20_000\\PSD_76_96_652.csv"
random_PSD_df = pd.read_csv(random_PSD_path)

# Create an instance of EEGFeatureExtractor
# extractor = EEGFeatureExtractor()
# extractor.calculate_spectral_skewness(random_PSD_df, normalize="tanh")
filter_instance = Filtering()
filter_instance.filter_eeg(1, 0.5, 30)

