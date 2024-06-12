import warnings
from typing import Callable, List

import numpy as np
import pandas as pd


def _find_features_to_group_ith(
    feature_table: pd.DataFrame, i: int, measure_func: Callable[[pd.Series], pd.Series]
) -> List[str]:
    sorted_values = measure_func(feature_table).sort_values()
    max_value_at_i = sorted_values.iloc[-i]
    cumulative_sum = np.cumsum(sorted_values)
    threshold_idx = np.argmax(cumulative_sum > max_value_at_i)
    features_to_group = sorted_values[:threshold_idx].index.tolist()
    return features_to_group


def find_features_to_group_by_abundance_ith(
    feature_table: pd.DataFrame, i: int
) -> List[str]:
    """
    Finds features to sum, such that the abundance of the summed features does
    not exceed the abundance of the i-th most abundant individual feature.
    """
    return _find_features_to_group_ith(feature_table, i, measure_func=lambda x: x.sum())


def find_features_to_group_by_variance_ith(
    feature_table: pd.DataFrame, i: int
) -> List[str]:
    """
    Finds features to sum, such that the variance of the summed features does
    not exceed the variance of the i-th most variant individual feature.
    """
    return _find_features_to_group_ith(feature_table, i, measure_func=lambda x: x.var())


def select_microbial_features(feat, method, i, ft_prefix):
    if method is None:
        # return original feature table
        return feat.copy()
    # if i larger than max index of feat.columns raise warning
    if i > len(feat.columns):
        warnings.warn(
            f"Selected i={i} is larger than number of "
            f"features. So it is set to the max. possible value: {len(feat.columns)}."
        )
        i = len(feat.columns)
    # todo: add warning when all features are grouped! --> verify when this
    # could happen
    if method == "abundance_ith":
        group_ft_ls = find_features_to_group_by_abundance_ith(feat, i)
        group_name = "_low_abun"
    elif method == "variance_ith":
        group_ft_ls = find_features_to_group_by_variance_ith(feat, i)
        group_name = "_low_var"
    else:
        raise ValueError(f"Unknown method: {method}.")

    if len(group_ft_ls) == 1:
        warnings.warn(
            f"No features found to group using method: {method}. "
            f"Returning original feature table."
        )
        return feat.copy()

    feat_selected = feat.copy()
    feat_selected[f"{ft_prefix}{group_name}"] = feat_selected[group_ft_ls].sum(axis=1)
    feat_selected.drop(columns=group_ft_ls, inplace=True)

    return feat_selected
