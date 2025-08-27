from MachineLearning.Evaluation.split_manager import SplitManager
from MachineLearning.IO.load_data import PathUtils
from MachineLearning.Utils.feature_utils import FeatureUtils


def create_single_split(hyperparameters, class_0, class_1, folderpath, test_size=0.15, random_state=61):
    # Combine without normalizing first
    features = ["bandpower", "spectral_skewness", "spectral_kurtosis", "shannon_entropy", "permutation_entropy"]
    FeatureUtils.combine_features(hyperparameters, "faw", False,features, normalize=False)
    FeatureUtils.combine_features(hyperparameters, "awake", False, features, normalize=False)


    split_manager = SplitManager(hyperparameters, class_0, class_1, test_size, random_state)
    split_manager.load_and_validate()
    train_df, test_df = split_manager.create_single_split()

    train_fullpath = PathUtils.return_anypath(folderpath, "train_set.parquet")
    test_fullpath = PathUtils.return_anypath(folderpath, "test_set.parquet")
    train_df.to_parquet(train_fullpath)
    test_df.to_parquet(test_fullpath)



if __name__ == "__main__":
    hyperparameters_ = {
        "merged_episodes": False,
        "bis_threshold": 70,
        "mac_threshold": 0.8,
        "min_episode_length": 20,
        "refractory_time": 5,
        "fixed_window_size": 20,
        "overlap": 0.0
    }
    class_0_ = "faw"
    class_1_ = "awake"
    folderpath_to_save = "D:\\Daten\\Other\\Splits_for_normalization_statistics\\"
    create_single_split(hyperparameters_, class_0_, class_1_, folderpath_to_save)