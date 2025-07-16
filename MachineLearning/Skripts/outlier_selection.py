from MachineLearning.Utils.config_handler import update_config
from MachineLearning.Evaluation.meta_fold_analyzer import MetaFoldAnalyzer

def select_multiple_outliers(overlaps_, min_episode_lengths_):
    for overlap in overlaps_:
        for ep_length in min_episode_lengths_:
            new_params = {
                "current_params": {
                    "overlap": overlap,
                    "min_episode_length": 20,
                    "fixed_window_size": ep_length
                }
            }
            current_params = update_config("parameters_config.yaml", new_params)["current_params"]
            fold_analyzer = MetaFoldAnalyzer("svm", current_params)
            outlier_df = fold_analyzer.select_outlier_groups(save_res=True, error_rate_threshold=0.5)
            print(outlier_df)


if __name__ == "__main__":
    overlaps = [0.0]
    min_episode_lengths = [15]

    select_multiple_outliers(overlaps, min_episode_lengths)