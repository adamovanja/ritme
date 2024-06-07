import warnings
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


def select_microbial_features(feat, method, idx_threshold, ft_prefix):
    # todo: finish this function - this is WIP!
    # if idx_threshold larger than max index of feat.columns raise warning
    if idx_threshold > len(feat.columns):
        Warning(
            f"Selected idx_threshold={idx_threshold} is larger than number of "
            f"features. So it is set to the max. possible value: {len(feat.columns)}."
        )
        idx_threshold = len(feat.columns)
    # todo: add warning when all features are grouped! --> verify when this
    # could happen
    if method == "abundance":
        group_ft_ls = find_features_to_group_by_abundance(feat, idx_threshold)
        group_name = "_low_abun"
    elif method == "variance":
        group_ft_ls = find_features_to_group_by_variance(feat, idx_threshold)
        group_name = "_low_var"
    else:
        raise ValueError(f"Unknown method: {method}")

    if len(group_ft_ls) == 1:
        warnings.warn(
            f"No features found to group using method: {method}. "
            f"Returning original feature table."
        )
        feat_selected = feat.copy()
    else:
        feat_selected = feat.copy()
        feat_selected[f"{ft_prefix}{group_name}"] = feat_selected[group_ft_ls].sum(
            axis=1
        )
        feat_selected.drop(columns=group_ft_ls, inplace=True)

    return feat_selected
