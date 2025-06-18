"""This Module contains a Utils class for test-train-split creation."""
import pandas as pd


class SplitUtils:
    """Helper class for splitting data into train and test sets."""

    @staticmethod
    def find_patient_split_by_epoch_balance(
            awake_df, faw_df, test_size=0.2, tolerance=0.05, max_iter=500, random_state=42
    ):
        import numpy as np
        import pandas as pd
        from sklearn.utils import shuffle

        rng = np.random.RandomState(random_state)

        # Alle ResultIDs, egal ob in awake oder faw
        all_ids = pd.Index(awake_df['ResultID'].unique()).union(faw_df['ResultID'].unique())

        # Mapping von ResultID → Anzahl Epochen
        id_episode_counts = {
            rid: len(awake_df[awake_df['ResultID'] == rid]) + len(faw_df[faw_df['ResultID'] == rid])
            for rid in all_ids
        }

        total_episodes = sum(id_episode_counts.values())
        target_test_episodes = int(test_size * total_episodes)
        best_diff = float('inf')

        best_train_ids = None
        best_test_ids = None

        for _ in range(max_iter):
            shuffled_ids = shuffle(all_ids, random_state=rng.randint(0, 99999))
            test_ids = []
            test_sum = 0

            for rid in shuffled_ids:
                count = id_episode_counts[rid]
                if test_sum + count > target_test_episodes:
                    continue
                test_ids.append(rid)
                test_sum += count

                # Prüfen ob innerhalb Toleranz
                rel_error = abs(test_sum - target_test_episodes) / target_test_episodes
                if rel_error <= tolerance:
                    break

            train_ids = [rid for rid in all_ids if rid not in test_ids]
            diff = abs(test_sum - target_test_episodes)

            if diff < best_diff:
                best_diff = diff
                best_test_ids = test_ids
                best_train_ids = train_ids

                if best_diff == 0 or rel_error <= tolerance:
                    break

        return best_train_ids, best_test_ids

    @staticmethod
    def balance_classes(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
        """
        Takes a feature set dataframe with a "label" column that only contain two classes. It balances the number
        of the rows to ensure, that there is the same amount of samples for each class, by randomly selecting from the
        larger class as many samples as in the smaller class.
        :param df: A dataframe with a "label" column and the classes 1 and 0.
        :param random_state: The random state to randomly select from the dataframe (for reproducibility).
        :return: The balanced dataframe.
        """
        class_1_df = df[df["label"] == 1]
        class_0_df = df[df["label"] == 0]

        # Get the size of the smaller class
        min_size = min(len(class_1_df), len(class_0_df))

        # Randomly select samples from larger class and select all from smaller class
        class_1_sampled = class_1_df.sample(n=min_size, random_state=random_state)
        class_0_sampled = class_0_df.sample(n=min_size, random_state=random_state)

        return pd.concat([class_1_sampled, class_0_sampled], ignore_index=True)

    @staticmethod
    def adjust_splits_to_ratio(test_df: pd.DataFrame, train_df: pd.DataFrame, test_ratio,
                               random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Adjust a train-test-split to a given ratio on sample level by adjusting the set, that is larger than the ratio
        and sampling it down. Function returns the new split and does nothing if the split is of optimal ratio.
        :param test_df: Dataframe with test set.
        :param train_df: Dataframe with train set.
        :param test_ratio: Given test ratio. Train ratio can be inferred by 1-test_ratio
        :param random_state: The random state to randomly select from the dataframe (for reproducibility).
        :return: Modified train and test dataframes as Tuple -> (new_test, new_train)
        """

        train_len = len(train_df)
        test_len = len(test_df)
        total_samples = train_len + test_len

        train_ratio = 1 - test_ratio

        # sample down test_set if ratio of test set higher than target test ratio and vice versa
        if test_len / total_samples > test_ratio:
            test_target_size = int((train_len * test_ratio) / train_ratio)
            test_df = test_df.sample(n=test_target_size, random_state=random_state)
        elif test_len / total_samples < test_ratio:
            train_target_size = int((test_len * train_ratio) / test_ratio)
            train_df = train_df.sample(n=train_target_size, random_state=random_state)

        return test_df, train_df
