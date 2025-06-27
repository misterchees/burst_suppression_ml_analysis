"""This Module contains a Utils class for test-train-split creation."""
from typing import Optional, Set, Any
import pandas as pd


class SplitUtils:
    """Helper class for splitting data into train and test sets."""

    @staticmethod
    def find_patient_split_by_epoch_balance(awake_df, faw_df, test_size: float,
                                            tolerance=0.05, max_iter=500, random_state=42,
                                            exclude_ids: Optional[Set[Any]] = None):
        """
        Splits awake and faw data into train and test sets along patient IDs. Tries to find the best split by preserving
        the ratio given by test_size as good as possible. Will throw an exception if the size of splittable samples
        is smaller than the target test sample size.

        :param awake_df: DataFrame containing awake data.
        :param faw_df: DataFrame containing fake awake data.
        :param test_size: Float between 0 and 1. Determines the ratio of the split.
        :param tolerance: Float between 0 and 1. How much the calculated split can diverge from given ratio.
        :param max_iter: Maximum number of iterations to find the best split.
        :param random_state: Random state for reproducibility.
        :param exclude_ids: Set of IDs to exclude from the split search
         (e.g. to avoid overlaps with previous folds in CV)
        :return: Tuple of train and test splits.
        """

        # All resultIDs in union of awake and faw (i.e. present in at least one on the dataframes)
        all_ids = pd.Index(awake_df['ResultID'].unique()).union(faw_df['ResultID'].unique())

        # Will remove IDs from Set in which the search for the best test split happens
        if exclude_ids is not None:
            ids_for_search = all_ids.difference(pd.Index(exclude_ids))
        else:
            ids_for_search = all_ids

        # Mapping of resultID → Number of epochs
        id_episode_counts = {
            rid: len(awake_df[awake_df['ResultID'] == rid]) + len(faw_df[faw_df['ResultID'] == rid])
            for rid in all_ids
        }

        total_episodes = sum(id_episode_counts.values())  # Whole sample size
        target_test_episodes = int(test_size * total_episodes)  # target test sample size
        available_episodes = sum(id_episode_counts[rid] for rid in ids_for_search)  # Episodes for search

        # check if searchable sample size is smaller than test sample size
        if available_episodes < target_test_episodes:
            raise ValueError(
                f"Not enough remaining samples to satisfy test_size. "
                f"Required: {target_test_episodes}, available: {available_episodes}"
            )

        # Iterate through possible random splits to find one, that satisfies all criteria
        best_test_ids = SplitUtils.find_best_ids_for_target_test_sample_size(
            ids_for_search=ids_for_search,
            target_test_size=target_test_episodes,
            tolerance=tolerance,
            id_episode_dict=id_episode_counts,
            random_state=random_state,
            max_iter=max_iter
        )

        best_train_ids = [rid for rid in all_ids if rid not in best_test_ids]

        return best_train_ids, best_test_ids

    @staticmethod
    def  find_best_ids_for_target_test_sample_size(ids_for_search, target_test_size: int, tolerance: float,
                                                  id_episode_dict: dict, random_state: int, max_iter: int):
        """
        Tries to find the best patient IDs that is as close as possible to target test sample size (within tolerance).
        :param ids_for_search: Set of IDs. Function will try to pick a subset that best fullfills the requirements.
        :param target_test_size: Number of test samples that need to be in the test subset within given tolerance.
        :param tolerance: Tolerance for deviaton from optimal target test size.
        :param id_episode_dict: Dictionary of IDs to sample size of that ID.
        :param random_state: Random state for reproducibility.
        :param max_iter: Maximum number of iterations to find the best subset of test IDs.
        :return: Set of patient IDs that contains number of samples as close as possible to target test sample size.
        """
        from sklearn.utils import shuffle
        import numpy as np

        # Initiate random number generator
        rng = np.random.RandomState(random_state)

        best_diff = float('inf')  # start with infinite as upper limit for deviaton from optimal test_size
        rel_error = float('inf')  # Relative error for comparison with tolerance down the line
        best_test_ids = None

        for _ in range(max_iter):
            # change order of IDs to search randomly
            shuffled_ids = shuffle(ids_for_search, random_state=rng.randint(0, 99999))
            test_ids = []
            test_sum = 0

            # Collect IDs until target test size of corresponding samples is reached
            for rid in shuffled_ids:
                count = id_episode_dict[rid]
                if test_sum + count > target_test_size:
                    continue
                test_ids.append(rid)
                test_sum += count

                # Check if deviaton from optimal size is in tolerance range. If True -> end search
                rel_error = abs(test_sum - target_test_size) / target_test_size
                if rel_error <= tolerance:
                    break

            # Calculate difference between size of found IDs and target size
            diff = abs(test_sum - target_test_size)

            # Save smallest difference and corresponding IDs
            if diff < best_diff:
                best_diff = diff
                best_test_ids = test_ids

                # End search if optimal sample size or defiation from optimal size is in tolerance range.
                if best_diff == 0 or rel_error <= tolerance:
                    break

        return best_test_ids

    @staticmethod
    def return_balanced_split(train_ids, test_ids, random_state: int, full_df: pd.DataFrame = None,
                              class_0_df: pd.DataFrame = None, class_1_df: pd.DataFrame = None, test_size: float = None
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        This method returns a tuple of the train and test splits based on the given ids. Both sets will then be balanced
        to ensure a ratio of 1:1 for both classes. As last optional step, the ratio of the split can be ensured, by
        sampling down the bigger set (in relation to the target ratio).

        :param train_ids: The ids of the train split.
        :param test_ids: The ids of the test split.
        :param random_state: The random state for balancing
        :param full_df: The full dataframe to be split. If None, the two class-dataframes will be used.
        :param class_0_df: The class 0 dataframe. If None, the full dataframe will be used.
        :param class_1_df: The class 1 dataframe. If None, the full dataframe will be used.
        :param test_size: The target ratio of the test split. If not None splits will be adjusted.
        :return: A tuple containing the train and test splits in the order (train, test).
        """
        if class_0_df is not None and class_1_df is not None:
            full_df = pd.concat([class_1_df.copy(), class_0_df.copy()], ignore_index=True)
        elif full_df is None:
            raise ValueError("Input dataframes missing. Either provied a full dataframe or both class dataframes.")

        # Split patient data
        train_df = full_df[full_df["ResultID"].isin(train_ids)].copy()  # Copy to modify df without risk
        test_df = full_df[full_df["ResultID"].isin(test_ids)].copy()

        # Balance classes in both sets
        train_df = SplitUtils.balance_classes(train_df, random_state)
        test_df = SplitUtils.balance_classes(test_df, random_state)

        # Correct for ratio, since previous steps might have changed it
        if test_size is not None:
            test_df, train_df = SplitUtils.adjust_splits_to_ratio(test_df, train_df, test_size, random_state)

        return train_df, test_df

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
    def return_train_test_sample_idx(train_ids, test_ids, random_state: int, full_df: pd.DataFrame, test_size: float):
        train_df, test_df = SplitUtils.return_balanced_split(
            train_ids=train_ids,
            test_ids=test_ids,
            random_state=random_state,
            full_df=full_df,
            test_size=test_size
        )

        # Retrieve sample indices of distribution

        train_idx = train_df["orig_index"].to_numpy()
        test_idx = test_df["orig_index"].to_numpy()

        print("\nData for current fold:\n")
        print(f"Train set size: {len(train_idx)}, Test set size: {len(test_idx)}")
        print(f"Class distribution in train: class label 1 = {sum(train_df.label == 1)},"
              f"class label 0 = {sum(train_df.label == 0)}")
        print(f"Class distribution in test:  class label 1 = {sum(test_df.label == 1)},"
              f" class label 0 = {sum(test_df.label == 0)}")


        return train_idx, test_idx

    @staticmethod
    def adjust_splits_to_ratio(test_df: pd.DataFrame, train_df: pd.DataFrame, test_ratio,
                               random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Adjust a train-test-split to a given ratio on sample level by adjusting the set, that is larger than the ratio
        and sampling it down. The Function returns the new split and does nothing if the split is of optimal ratio.
        :param test_df: Test set dataframe.
        :param train_df: Train set dataframe.
        :param test_ratio: Given test ratio. Train ratio can be inferred by 1-test_ratio
        :param random_state: The random state to randomly select from the dataframe (for reproducibility).
        :return: Modified train and test dataframes as Tuple -> (new_test, new_train)
        """

        train_len = len(train_df)
        test_len = len(test_df)
        total_samples = train_len + test_len

        train_ratio = 1 - test_ratio

        # sample down test_set if the ratio of test set is higher than target test ratio and vice versa
        if test_len / total_samples > test_ratio:
            test_target_size = int((train_len * test_ratio) / train_ratio)
            test_df = test_df.sample(n=test_target_size, random_state=random_state)
        elif test_len / total_samples < test_ratio:
            train_target_size = int((test_len * train_ratio) / test_ratio)
            train_df = train_df.sample(n=train_target_size, random_state=random_state)

        return test_df, train_df

    @staticmethod
    def create_full_df(class_1_df: pd.DataFrame, class_0_df: pd.DataFrame) -> pd.DataFrame:
        full_df = pd.concat([class_1_df.copy(), class_0_df.copy()], ignore_index=True)
        full_df["label"] = full_df["ResultID"].map(
            lambda rid: 1 if rid in set(class_1_df["ResultID"]) else 0
        )
        # Save original full_df index to easily retrieve this information later
        full_df = full_df.reset_index().rename(columns={"index": "orig_index"})

        return full_df

    @staticmethod
    def return_X_y(full_df: pd.DataFrame) -> tuple:
        X = full_df.drop(columns=["label", "orig_index"])
        y = full_df["label"].values

        return X, y