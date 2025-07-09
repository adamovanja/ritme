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


def _find_features_to_group_topi(
    feature_table: pd.DataFrame,
    i: int,
    measure_func: Callable[[pd.DataFrame], pd.Series],
) -> List[str]:
    """
    Helper function to find features to group, excluding the top i features
    based on the given measure. Returns the remaining features as a list.
    """
    sorted_values = measure_func(feature_table).sort_values(ascending=False)
    features_to_group = sorted_values.index[i:].tolist()
    return features_to_group


def find_features_to_group_by_abundance_topi(
    feature_table: pd.DataFrame, i: int
) -> List[str]:
    """
    Finds features to sum, excluding the top i most abundant features. Returns
    the remaining features as a list.
    """
    return _find_features_to_group_topi(
        feature_table, i, measure_func=lambda x: x.sum()
    )


def find_features_to_group_by_variance_topi(
    feature_table: pd.DataFrame, i: int
) -> List[str]:
    """
    Finds features to sum, excluding the top i most variant features. Returns
    the remaining features as a list.
    """
    return _find_features_to_group_topi(
        feature_table, i, measure_func=lambda x: x.var()
    )


def _find_features_to_group_quantile(
    feature_table: pd.DataFrame,
    q: float,
    measure_func: Callable[[pd.DataFrame], pd.Series],
) -> List[str]:
    """
    Helper function to find features to group based on the specified quantile of
    the given measure. Returns the features below the quantile as a list.
    """
    measure_values = measure_func(feature_table)
    quantile_value = measure_values.quantile(q)
    features_to_group = measure_values[measure_values < quantile_value].index.tolist()
    return features_to_group


def find_features_to_group_by_abundance_quantile(
    feature_table: pd.DataFrame, q: float
) -> List[str]:
    """
    Finds features to sum based on the quantile of abundance. Returns the
    features with abundance lower than the specified quantile.
    """
    return _find_features_to_group_quantile(
        feature_table, q, measure_func=lambda x: x.sum()
    )


def find_features_to_group_by_variance_quantile(
    feature_table: pd.DataFrame, q: float
) -> List[str]:
    """
    Finds features to sum based on the quantile of variance. Returns the
    features with variance lower than the specified quantile.
    """
    return _find_features_to_group_quantile(
        feature_table, q, measure_func=lambda x: x.var()
    )


def _find_features_to_group_threshold(
    feature_table: pd.DataFrame,
    threshold: float,
    measure_func: Callable[[pd.DataFrame], pd.Series],
) -> List[str]:
    measure_values = measure_func(feature_table)
    features_to_group = measure_values[measure_values < threshold].index.tolist()
    return features_to_group


def find_features_to_group_by_abundance_threshold(
    feature_table: pd.DataFrame, threshold: float
) -> List[str]:
    """
    Finds features to sum based on a fixed threshold of abundance. Returns the
    features with abundance lower than the specified threshold.
    """
    return _find_features_to_group_threshold(
        feature_table, threshold, measure_func=lambda x: x.sum()
    )


def find_features_to_group_by_variance_threshold(
    feature_table: pd.DataFrame, threshold: float
) -> List[str]:
    """
    Finds features to sum based on a fixed threshold of variance. Returns the
    features with variance lower than the specified threshold.
    """
    return _find_features_to_group_threshold(
        feature_table, threshold, measure_func=lambda x: x.var()
    )


def _reset_i_too_large(i: int, feat: pd.DataFrame):
    if i > len(feat.columns):
        warnings.warn(
            f"Selected i={i} is larger than number of "
            f"features. So it is set to the max. possible value: {len(feat.columns)}."
        )
        return len(feat.columns)
    return i


def select_microbial_features(feat, config, ft_prefix):
    method = config["data_selection"]

    if method is None:
        # return original feature table
        return feat.copy()

    group_name = "_low_abun" if method.startswith("abundance") else "_low_var"

    if method == "abundance_ith":
        group_ft_ls = find_features_to_group_by_abundance_ith(
            feat, config["data_selection_i"]
        )
    elif method == "variance_ith":
        group_ft_ls = find_features_to_group_by_variance_ith(
            feat, config["data_selection_i"]
        )
    elif method == "abundance_topi":
        group_ft_ls = find_features_to_group_by_abundance_topi(
            feat, config["data_selection_i"]
        )
    elif method == "variance_topi":
        group_ft_ls = find_features_to_group_by_variance_topi(
            feat, config["data_selection_i"]
        )
    elif method == "abundance_quantile":
        group_ft_ls = find_features_to_group_by_abundance_quantile(
            feat, config["data_selection_q"]
        )
    elif method == "variance_quantile":
        group_ft_ls = find_features_to_group_by_variance_quantile(
            feat, config["data_selection_q"]
        )
    elif method == "abundance_threshold":
        group_ft_ls = find_features_to_group_by_abundance_threshold(
            feat, config["data_selection_t"]
        )
    elif method == "variance_threshold":
        group_ft_ls = find_features_to_group_by_variance_threshold(
            feat, config["data_selection_t"]
        )
    else:
        raise ValueError(f"Unknown method: {method}.")

    if len(group_ft_ls) == 1:
        warnings.warn(
            f"No features found to group using method: {method}. "
            f"Returning original feature table."
        )
        return feat.copy()
    elif len(group_ft_ls) == len(feat.columns):
        # this shouldn't be possible - if it does it needs to be fixed
        raise ValueError(
            f"All features are grouped using method: {method}. "
            f"This is not allowed. Please report this as an issue to "
            f"the ritme repos."
        )
    feat_selected = feat.copy()
    feat_selected[f"{ft_prefix}{group_name}"] = feat_selected[group_ft_ls].sum(axis=1)
    feat_selected.drop(columns=group_ft_ls, inplace=True)

    return feat_selected
