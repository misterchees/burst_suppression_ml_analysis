from MachineLearning.Evaluation.metrics_evaluator import MetricsEvaluator
from MachineLearning.IO.load_data import LoadData
from MachineLearning.Utils.config_handler import update_config


def print_metrics(overlaps_, min_episode_lengths_):
    loader = LoadData()
    evaluator = MetricsEvaluator(None, None, None, None, None)

    for overlap in overlaps_:
        for ep_length in min_episode_lengths_:
            new_params = {
                "current_params": {
                    "overlap": overlap,
                    "min_episode_length": 20,
                    "fixed_window_size": ep_length
                }
            }
            curent_params = update_config("parameters_config.yaml", new_params)["current_params"]
            print(f"\n#######Testing Parameters: {curent_params}\n")
            current_metrics = loader.load_metrics(curent_params, "svm")
            evaluator.print_result(current_metrics["summary"], True)


if __name__ == "__main__":
    overlaps = [0.25, 0.5, 0.75, 0.9]
    min_episode_lengths = [20]

    print_metrics(overlaps, min_episode_lengths)