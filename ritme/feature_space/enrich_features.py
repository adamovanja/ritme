import numpy as np
import pandas as pd

from ritme.feature_space.transform_features import PSEUDOCOUNT


def compute_shannon_diversity(
    df: pd.DataFrame, microbial_features: list, pseudocount: float = PSEUDOCOUNT
) -> pd.Series:
    """
    Compute Shannon diversity for each row over the given microbial features.
    """
    p = df[microbial_features].values
    entropy = -np.nansum(p * np.log(p + pseudocount), axis=1)
    return pd.Series(entropy, index=df.index)


def enrich_features(
    train_val_raw: pd.DataFrame,
    microbial_ft_ls: list,
    train_val_t: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, list]:
    """
    Enrich the feature table with additional features based on the configuration.

    Returns:
        enriched DataFrame, list of added feature names
    """
    other_ft_ls: list = []
    train_val_out = train_val_t.copy()

    if config["data_enrich"] is None:
        return other_ft_ls, train_val_out

    if config["data_enrich"] in ["shannon", "shannon_and_metadata"]:
        # add shannon entropy
        other_ft_ls.append("shannon_entropy")
        train_val_out["shannon_entropy"] = compute_shannon_diversity(
            train_val_raw,
            microbial_ft_ls,
        )

    if config["data_enrich"] in ["shannon_and_metadata", "metadata_only"]:
        # add metadata features
        missing_md = set(config["data_enrich_with"]) - set(train_val_raw.columns)
        if len(missing_md) > 0:
            raise ValueError(
                f"Feature {missing_md} not found in the "
                "training data. Please update the experiment configuration entry "
                "'data_enrich_with'."
            )
        else:
            # hot encode if config["data_enrich_with"] is a categorical else
            # just append to list of features to use for modelling
            for feat in config["data_enrich_with"]:
                col = train_val_raw[feat]
                if isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object:
                    dummies = pd.get_dummies(
                        col, prefix=feat, drop_first=True, dtype=float
                    )
                    train_val_out = pd.concat([train_val_out, dummies], axis=1)
                    other_ft_ls.extend(dummies.columns.tolist())
                else:
                    other_ft_ls.append(feat)

    return other_ft_ls, train_val_out
