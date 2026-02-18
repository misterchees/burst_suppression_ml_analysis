from MachineLearning.Utils.config_handler import update_config
from MachineLearning.Evaluation.meta_fold_analyzer import MetaFoldAnalyzer


def select_multiple_outliers(_new_params, _model_key, print_outliers=True, outlier_run_name=None):
    """Selects multiple outlier groups based on the given configuration, analyzing models and parameters"""
    all_params = update_config("parameters_config.yaml", _new_params)
    fold_analyzer = MetaFoldAnalyzer(_model_key, all_params["current_params"], outlier_run_name)
    outlier_df = fold_analyzer.select_outlier_groups(
        save_res=True,
        error_rate_threshold=0.5
    )

    if print_outliers:
        print(outlier_df)
    return outlier_df


if __name__ == "__main__":
    model_key = "svm"

    overlap = 0.0
    min_episode_length = 20

    new_params = {
        "current_params": {
            "overlap": overlap,
            "min_episode_length": 20,
            "fixed_window_size": min_episode_length
        }
    }

    select_multiple_outliers(new_params, model_key)
