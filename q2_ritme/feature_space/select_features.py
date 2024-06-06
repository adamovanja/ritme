from typing import Callable, List

import numpy as np
import pandas as pd


def _find_features_to_group(
    feature_table: pd.DataFrame, i: int, measure_func: Callable[[pd.Series], pd.Series]
) -> List[str]:
    sorted_values = measure_func(feature_table).sort_values()
    max_value_at_i = sorted_values.iloc[-i]
    cumulative_sum = np.cumsum(sorted_values)
    threshold_idx = np.argmax(cumulative_sum > max_value_at_i)
    features_to_group = sorted_values[:threshold_idx].index.tolist()
    return features_to_group


def find_features_to_group_by_abundance(
    feature_table: pd.DataFrame, i: int
) -> List[str]:
    """
    Finds features to sum, such that the abundance of the summed features does
    not exceed the abundance of the i-th most abundant individual feature.
    """
    return _find_features_to_group(feature_table, i, measure_func=lambda x: x.sum())


def find_features_to_group_by_variance(
    feature_table: pd.DataFrame, i: int
) -> List[str]:
    """
    Finds features to sum, such that the variance of the summed features does
    not exceed the variance of the i-th most variant individual feature.
    """
    return _find_features_to_group(feature_table, i, measure_func=lambda x: x.var())


def select_microbial_features(feat, method, method_i):
    # todo: finish this function - this is WIP!
    # if method_i larger than max index of feat.columns raise warning
    if method_i > len(feat.columns):
        Warning(
            f"Selected method_i={method_i} is larger than number of features. So it is "
            f"set to the max. possible value: {len(feat.columns)}."
        )
        method_i = len(feat.columns)

    if method == "abundance":
        group_ft_ls = find_features_to_group_by_abundance(feat, method_i)
    elif method == "variance":
        group_ft_ls = find_features_to_group_by_variance(feat, method_i)
    else:
        raise ValueError(f"Unknown method: {method}")

    if len(group_ft_ls) == 1:
        raise ValueError(f"No features to group found using method: {method}")

    grouped_feat = feat[group_ft_ls].sum(axis=1)
    # todo: append grouped features to non-grouped features
    # todo: add new names to grouped features
    return grouped_feat
