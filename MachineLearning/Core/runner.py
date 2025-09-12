"""
This module is to execute all pipeline commands and therefore the main point to run the project.
"""
from MachineLearning.Core.pipeline import Pipeline
from MachineLearning.Utils.config_handler import replace_bands_in_config
import itertools

INITIAL_DATA_SUBDIR_KEY = "combined_raw_data"
channel = 1
model_key = "svm"
model_params = {"C": 1, "kernel": "rbf"}
filter_method = "butterworth"
normalize_method = "zscore"
transform_method = "welch"
run_name = "norm2_z_score_0"

all_run_params_dict = {
    "current_params": {
        "merged_episodes": False,
        "bis_threshold": 70,
        "mac_threshold": 0.8,
        "min_episode_length": 20,
        "refractory_time": 5,
        "fixed_window_size": 20,
        "overlap": 0.0
    },
    "filtering_params": {
        filter_method: {"lowcut": 0.5, "highcut": 30.0, "order": 4}
    },
    "normalizing_params": {
        normalize_method: normalize_method
    },
    "transform_params": {
        transform_method: {"channel": channel, "nperseg_seconds": 2, "fs": 128}
    },
    "feature_params": {
        "relative_bandpower": {"normalize_to": "bands"},
        "shannon_entropy": {"normalize": False},
        "spectral_skewness": {"normalize": False, "n_method": "clip", "lower_bound": 0, "upper_bound": 1},
        "spectral_kurtosis": {"normalize": False, "n_method": "clip", "lower_bound": 0, "upper_bound": 1},
        "mean": {"channel": channel},
        "variance": {"channel": channel},
        "amplitude": {"channel": channel},
        "sample_entropy": {"channel": channel, "emb_dim": 2, "tolerance": 0.2},
        "permutation_entropy": {"channel": channel, "order": 3, "delay": 1, "normalize": False},
        "fuzzy_entropy": {"channel": channel, "m": 2, "r": 0.2, "n": 2}
    },
    "classification_params": {
        "test_size": 0.15,
        "random_seed": 42,
        "remove_outliers": False,
        "remove_outlier_epochs": False,
        "outlier_run_name": "norm2_in_place_rm_outlier_5",
        model_key: model_params
    }
}

# Use None for variable to skip step; Use "all_features" if all features should be used in step
features = ["bandpower", "spectral_skewness", "spectral_kurtosis", "shannon_entropy", "permutation_entropy"]
features_to_combine = ["bandpower", "spectral_skewness", "spectral_kurtosis", "shannon_entropy", "permutation_entropy"]
band_dict = {'Delta': [0.5, 4], 'Theta': [4, 8], 'Alpha': [8, 13], 'Beta': [13, 30]}

bands_to_remove = []

# Set dict to None if no extraction AND no combination shall be conduced
features_dict = {
    "features": features,
    "features_to_combine": features_to_combine
}

# Metadata to analyze for errors
metadata_to_analyze = ["ResultID"]

epoch_classes = {0: "faw", 1: "awake"}  # Actual ML Project
# epoch_classes = {0: "normal_an", 1: "awake"}  # Sanity Check

steps_of_workflow = ["combine"]


def run():
    """
    Function to execute any code
    """
    # Set which bands to keep from bandpower
    band_dict_for_iteration = {band: f_range for band, f_range in band_dict.items() if band not in bands_to_remove}
    replace_bands_in_config("parameters_config.yaml",
                            {"feature_params":
                       {"relative_bandpower":
                            {"frequency_bands": band_dict_for_iteration}}})

    pipeline = Pipeline(
        init_data_key=INITIAL_DATA_SUBDIR_KEY,
        epoch_classes=epoch_classes,
        update_dict=all_run_params_dict,
        filter_method=filter_method,
        model_key=model_key,
        normalize_method=normalize_method,
        transform_method=transform_method,
        features_dict=features_dict,
        metadata_to_analyze=metadata_to_analyze,
        run_name=run_name,
        force_overwrite=True,
        force_transform=False,
        force_extract=True,
        global_outliers=False
    )

    pipeline.complete_run(steps_of_workflow)
    # Setting original bands back
    replace_bands_in_config("parameters_config.yaml",
                            {"feature_params":
                       {"relative_bandpower":
                            {"frequency_bands": band_dict}}})


def generate_feature_combinations(base_features=("bandpower", "spectral_skewness", "spectral_kurtosis", "shannon_entropy", "permutation_entropy"),
                                  all_bands=("Alpha", "Beta", "Theta", "Delta")):
    """
    Generate all possible feature and bandpower combinations with compact, sorted run names.

    :param base_features: List of features (incl. "bandpower" if wanted).
    :param all_bands: List of all available bands for bandpower.
    :returns: List of (features_to_combine, bands_to_remove_, run_name_).
    """
    feature_short = {
        "bandpower": "band",
        "spectral_skewness": "skew",
        "spectral_kurtosis": "kurt",
        "shannon_entropy": "shan",
        "permutation_entropy": "perm"
    }

    runs = []

    # All subsets of features
    for feature_subset_size in range(1, len(base_features) + 1):
        for feature_subset in itertools.combinations(base_features, feature_subset_size):
            feature_subset = list(feature_subset)

            if "bandpower" in feature_subset:
                for band_subset_size in range(1, len(all_bands) + 1):
                    for band_subset in itertools.combinations(all_bands, band_subset_size):
                        bands_to_remove_ = [b for b in all_bands if b not in band_subset]

                        band_short = "".join(sorted([b[0].upper() for b in band_subset]))
                        feature_names = []
                        for f in feature_subset:
                            if f == "bandpower":
                                feature_names.append(f"band{band_short}")
                            else:
                                feature_names.append(feature_short[f])

                        # Every run_name_ has the same order of abbreviations
                        feature_names = sorted(feature_names)
                        run_name_ = "zscorenorm_" + "_".join(feature_names)

                        runs.append((feature_subset, bands_to_remove_, run_name_))
            else:
                feature_names = [feature_short[f] for f in feature_subset]
                feature_names = sorted(feature_names)
                run_name_ = "zscorenorm_" + "_".join(feature_names)
                runs.append((feature_subset, [], run_name_))

    return runs


if __name__ == "__main__":
    run()
    # run_combinations = generate_feature_combinations()
    # for run_combination in run_combinations:
    #     features_to_combine, bands_to_remove, run_name = run_combination
    #     features_dict["features_to_combine"] = features_to_combine
    #     run()
