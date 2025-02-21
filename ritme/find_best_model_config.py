import json
import os

import pandas as pd
import qiime2 as q2
import skbio
import typer
from qiime2.plugins import phylogeny

from ritme._decorators import helper_function, main_function
from ritme.evaluate_models import (
    TunedModel,
    retrieve_n_init_best_models,
    save_best_models,
)
from ritme.tune_models import run_all_trials


# ----------------------------------------------------------------------------
@helper_function
def _load_experiment_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


@helper_function
def _verify_experiment_config(config: dict):
    """
    Raises valueError if provided tracking_uri in config is not 'mlruns' or
    'wandb'.
    """
    if config["tracking_uri"] not in ["mlruns", "wandb"]:
        raise ValueError(
            f"Invalid tracking_uri: {config['tracking_uri']}. Must be "
            f"'mlruns' or 'wandb'."
        )


@helper_function
def _save_config(config: dict, path_output: str, file_name: str):
    """Save configuration"""
    file_path = os.path.join(path_output, file_name)

    with open(file_path, "w") as f:
        return json.dump(config, f)


@helper_function
def _load_taxonomy(path_to_tax: str) -> pd.DataFrame:
    """Load taxonomy data"""
    art_taxonomy = q2.Artifact.load(path_to_tax)
    return art_taxonomy.view(pd.DataFrame)


@helper_function
def _process_taxonomy(tax: pd.DataFrame, ft: pd.DataFrame) -> pd.DataFrame:
    """Process taxonomy data"""
    df_tax = tax.copy()
    # rename taxonomy to match "F" feature names
    df_tax.index = df_tax.index.map(lambda x: "F" + str(x))

    # filter the taxonomy based on the feature table
    df_tax_f = df_tax[df_tax.index.isin(ft.columns.tolist())]

    if df_tax_f.shape[0] == 0:
        raise ValueError("Taxonomy data does not match with feature table.")

    return df_tax_f


@helper_function
def _load_phylogeny(path_to_phylo: str) -> skbio.TreeNode:
    art_phylo = q2.Artifact.load(path_to_phylo)
    return art_phylo.view(skbio.TreeNode)


@helper_function
def _process_phylogeny(phylo_tree: skbio.TreeNode, ft: pd.DataFrame) -> skbio.TreeNode:
    """Process phylogeny"""
    # filter tree by feature table: this prunes a phylogenetic tree to match
    # the input ids
    # Remove the first letter of each column name: "F" to match phylotree
    ft_i = ft.copy()
    ft_i.columns = [col[1:] for col in ft_i.columns]
    art_ft_i = q2.Artifact.import_data("FeatureTable[RelativeFrequency]", ft_i)

    art_phylo = q2.Artifact.import_data("Phylogeny[Rooted]", phylo_tree)
    (art_phylo_f,) = phylogeny.actions.filter_tree(tree=art_phylo, table=art_ft_i)
    tree_phylo_f = art_phylo_f.view(skbio.TreeNode)

    # add prefix "F" to leaf names in tree to remain consistent with ft
    for node in tree_phylo_f.tips():
        node.name = "F" + node.name

    # ensure that # leaves in tree == feature table dimension
    num_leaves = tree_phylo_f.count(tips=True)
    assert num_leaves == ft.shape[1]

    return tree_phylo_f


@helper_function
def _define_model_tracker(tracking_uri: str, path_store_model_logs: str) -> str:
    if tracking_uri == "mlruns":
        path_tracker = os.path.join(path_store_model_logs, tracking_uri)
        print(
            f"You can view the model logs by launching MLflow UI from within folder "
            f": {path_store_model_logs}."
        )
    else:
        path_tracker = "wandb"
        print("You can view the model logs by logging into your wandb account.")
    return path_tracker


@helper_function
def _define_experiment_path(config, path_store_model_logs):
    # path to experiments and their logs
    path_exp = os.path.join(path_store_model_logs, config["experiment_tag"])
    if os.path.exists(path_exp):
        raise ValueError(
            f"This experiment tag already exists: {config['experiment_tag']}."
            "Please use another one."
        )
    else:
        os.makedirs(path_exp)
    return path_exp


# ----------------------------------------------------------------------------
@main_function
def find_best_model_config(
    config: dict,
    train_val: pd.DataFrame,
    tax: pd.DataFrame = None,
    tree_phylo: skbio.TreeNode = None,
    path_store_model_logs: str = "experiments/models",
) -> tuple[dict[str, TunedModel], str]:
    """
    Find the best model configuration incl. feature representation.

    Args:
        config (dict): Model configuration space.
        train_val (pd.DataFrame): Dataset to train and validate models.
        tax (pd.DataFrame, optional): Taxonomy matching features starting with
        'F' in `train_val`. Needed for training trac models and feature
        engineering based on taxonomy. Defaults to None.
        tree_phylo (skbio.TreeNode, optional): Phylogenetic tree for features
        starting with "F" in `train_val`. Needed for training trac models.
        Defaults to None.
        path_store_model_logs (str, optional): Path to store model logs.
        Defaults to "experiments/models".

    Raises:
        ValueError: If the provided experiment tag in config already exists in
        `path_store_model_logs`.

    Returns:
        dict: With model type as key and best TunedModel as value.
        str: Path to the experiment folder.
    """
    _verify_experiment_config(config)

    # ! Define needed paths
    path_exp = _define_experiment_path(config, path_store_model_logs)
    _save_config(config, path_exp, "experiment_config.json")

    # define model tracker
    path_tracker = _define_model_tracker(config["tracking_uri"], path_store_model_logs)

    # ! Process taxonomy and phylogeny by microbial feature table
    ft_col = [x for x in train_val.columns if x.startswith("F")]
    if tax is not None:
        tax = _process_taxonomy(tax, train_val[ft_col])
    if tree_phylo is not None:
        tree_phylo = _process_phylogeny(tree_phylo, train_val[ft_col])

    # ! Run all experiments on train_val
    result_dic = run_all_trials(
        train_val,
        config["target"],
        config["group_by_column"],
        config["seed_data"],
        config["seed_model"],
        tax,
        tree_phylo,
        path_tracker,
        path_exp,
        # number of trials to run per model type * grid_search parameters in
        # @_static_searchspace
        config["num_trials"],
        config["max_cuncurrent_trials"],
        model_types=config["ls_model_types"],
        fully_reproducible=False,
        test_mode=config["test_mode"],
        model_hyperparameters=config.get("model_hyperparameters", {}),
    )

    # ! Get best models of this experiment
    best_model_dic = retrieve_n_init_best_models(result_dic, train_val)

    return best_model_dic, path_exp


@main_function
def cli_find_best_model_config(
    path_to_config: str,
    path_to_train_val: str,
    path_to_tax: str = None,
    path_to_tree_phylo: str = None,
    path_store_model_logs: str = "experiments/models",
):
    """
    Find the best model configuration incl. feature representation.

    Args:
        path_to_config (str): Path to experiment configuration file.
        path_to_train_val (str): Path to train_val dataset.
        path_to_tax (str, optional): Path to taxonomy QIIME2 artifact of type
        FeatureData[Taxonomy] matching features starting with 'F' in
        `train_val`. Needed for training trac models and feature engineering
        based on taxonomy. Defaults to None.
        path_to_tree_phylo (str, optional): Path to phylogenetic tree QIIME2
        artifact of type "Phylogeny[Rooted]" for features starting with "F" in
        `train_val`. Needed for training trac models. Defaults to None.
        path_store_model_logs (str, optional): Path to store model logs.
        Defaults to "experiments/models".

    Raises:
        ValueError: If the provided experiment tag in config already exists in
        `path_store_model_logs`.

    Side Effects:
        Writes the best tuned model configurations to a file in the specified
        experiment path in path_to_config.
    """
    config = _load_experiment_config(path_to_config)
    train_val = pd.read_pickle(path_to_train_val)

    if path_to_tax is not None:
        tax = _load_taxonomy(path_to_tax)
    else:
        tax = None

    if path_to_tree_phylo is not None:
        tree_phylo = _load_phylogeny(path_to_tree_phylo)
    else:
        tree_phylo = None

    best_model_dict, path_exp = find_best_model_config(
        config, train_val, tax, tree_phylo, path_store_model_logs
    )

    # save each best tuned model
    save_best_models(best_model_dict, path_exp)
    print(f"Best model configurations were saved in {path_exp}.")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    typer.run(cli_find_best_model_config)
