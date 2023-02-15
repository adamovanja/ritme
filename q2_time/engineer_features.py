import pandas as pd
from skbio.stats.composition import clr

PSEUDOCOUNT = 0.0001


def transform_features(
    feat: pd.DataFrame, method: str, pseudocount: float = PSEUDOCOUNT
) -> pd.DataFrame:
    """Transform features with specified method"""
    # todo: create processing pipeline see:
    # see https://scikit-learn.org/stable/data_transforms.html
    if method == "clr":
        feat.replace(0, pseudocount, inplace=True)

        feat_trans = pd.DataFrame(clr(feat), index=feat.index, columns=feat.columns)
    else:
        raise ValueError(f"Method {method} is not implemented yet.")

    return feat_trans
