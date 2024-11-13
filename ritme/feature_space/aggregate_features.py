import warnings

import pandas as pd

TAX_RANKS = [
    "domain",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]


def extract_taxonomic_entity(taxonomy_df: pd.DataFrame, tax_rank: str) -> dict:
    """
    Extracts the taxonomic entity from a taxonomy DataFrame. If this tax_rank is
    not provided it is filled with an unknown value.

    Args:
        taxonomy_df (pd.DataFrame): A pandas DataFrame containing taxonomy
        information with a 'Taxon' column.
        tax_rank (str): The taxonomic rank to extract (e.g. "species")

    Returns:
        dict: A dictionary with OTU ids as keys and respective taxonomic
        entities as values.
    """
    taxonomy_refined = taxonomy_df.copy()
    tax_ranks_dic = {rank: index for index, rank in enumerate(TAX_RANKS)}
    idx = tax_ranks_dic[tax_rank]
    prefix = tax_rank[0] + "__"

    taxonomy_refined[tax_rank] = (
        taxonomy_refined["Taxon"].str.split(";").str[idx].str.strip()
    )
    # remove unnecessary [] characters
    for char in [r"\[", r"\]"]:
        taxonomy_refined[tax_rank] = taxonomy_refined[tax_rank].str.replace(
            char, "", regex=True
        )

    # flag unknown values at this level
    taxonomy_refined[tax_rank] = taxonomy_refined[tax_rank].replace(
        prefix, f"{prefix}unknown"
    )
    taxonomy_refined[tax_rank] = taxonomy_refined[tax_rank].fillna(f"{prefix}unknown")

    return taxonomy_refined[tax_rank].to_dict()


def aggregate_ft_by_taxonomy(
    feature_table_df: pd.DataFrame, tax_dict: dict
) -> pd.DataFrame:
    """
    Aggregates a feature table DataFrame by taxonomic order in order_dict.

    Args:
        feature_table_df (pd.DataFrame): A pandas DataFrame containing the
        feature table with OTU ids as index.
        tax_dict (dict): A dictionary with OTU ids as keys and taxonomic orders
        as values.

    Returns:
        pd.DataFrame: A pandas DataFrame with the aggregated feature table
        grouped by taxonomic order.
    """
    # Create a DataFrame with OTU and corresponding taxonomic entity
    tax_df = pd.DataFrame.from_dict(tax_dict, orient="index", columns=["tax_order"])
    tax_df.index.name = "OTU"

    # Merge the feature table with the order DataFrame
    ft_not_in_tax = [
        x for x in feature_table_df.columns.tolist() if x not in tax_df.index.tolist()
    ]
    if len(ft_not_in_tax) > 0:
        warnings.warn(
            f"These features were not found in taxonomy and are hence "
            f"disregarded: {ft_not_in_tax}."
        )
    merged_df = feature_table_df.T.merge(tax_df, left_index=True, right_index=True)

    # Group the merged DataFrame by the tax_order column and sum the rel abundances
    aggregated_df = merged_df.groupby("tax_order").sum()
    aggregated_df = aggregated_df.T
    aggregated_df.columns.name = None
    aggregated_df.index.name = "id"
    # sorted alphabetically

    return aggregated_df


def agg_microbial_fts_taxonomy(
    ft: pd.DataFrame, tax_entity: str, df_taxonomy: pd.DataFrame
):
    """
    Aggregates feature table by taxonomic entity. If this taxonomic entity is
    unknown it uses the next higher rank.

    Args:
        ft (pd.DataFrame): The feature table to transform.
        tax_entity (str): The taxonomic entity to use for transformation.
        df_taxonomy (pd.DataFrame): The taxonomy DataFrame.

    Returns:
        pd.DataFrame: The transformed feature table.
    """
    # get taxonomic entities: dic[OTU] = tax_entity at level of interest only
    tax_dict = extract_taxonomic_entity(df_taxonomy, tax_entity)

    # replace __unknowns with tax. entity of higher rank
    tax_ranks_dic = {rank: index for index, rank in enumerate(TAX_RANKS)}
    idx = tax_ranks_dic[tax_entity]
    refine_unknown = TAX_RANKS[idx - 1]

    unknown_dict = extract_taxonomic_entity(df_taxonomy, refine_unknown)
    for key, value in tax_dict.items():
        tax_unknown = f"{tax_entity[0]}__unknown"
        if value == tax_unknown:
            tax_dict[key] = f"{tax_unknown[:-3]}_{unknown_dict[key]}"

    # aggregate features by derived taxonomic entities
    ft_aggregated = aggregate_ft_by_taxonomy(ft, tax_dict)

    return ft_aggregated


def aggregate_microbial_features(
    feat: pd.DataFrame,
    method: str,
    df_taxonomy: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate features with specified `method`.
    """
    method_map = {
        "tax_class": agg_microbial_fts_taxonomy,
        "tax_order": agg_microbial_fts_taxonomy,
        "tax_family": agg_microbial_fts_taxonomy,
        "tax_genus": agg_microbial_fts_taxonomy,
        None: None,
    }
    if method not in method_map.keys():
        raise ValueError(f"Method {method} is not implemented yet.")

    # aggregate
    if method is not None and method.startswith("tax_"):
        tax_entity = method.split("_")[1]
        feat_agg = method_map[method](feat, tax_entity, df_taxonomy)
    else:
        feat_agg = feat.copy()
    return feat_agg
