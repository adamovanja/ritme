import json
import os
import tempfile

import pandas as pd
import skbio
import typer

from ritme._decorators import helper_function, main_function
from ritme.evaluate_models import (
    TunedModel,
    retrieve_n_init_best_models,
    save_best_models,
)
from ritme.feature_space.utils import _PAST_SUFFIX_RE
from ritme.split_train_test import adaptive_k_folds
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
    """Load taxonomy from a tab-delimited file (.tsv).

    Expected columns: Feature ID (index), Taxon, Confidence (optional).
    """
    return pd.read_csv(path_to_tax, sep="\t", index_col=0)


@helper_function
def _process_taxonomy(tax: pd.DataFrame, ft: pd.DataFrame) -> pd.DataFrame:
    """Process taxonomy data"""
    df_tax = tax.copy()
    # rename taxonomy to match "F" feature names
    df_tax.index = df_tax.index.map(lambda x: "F" + str(x))

    # Reference the t0 snapshot feature set (unsuffixed columns)
    ft_cols = ft.columns.tolist()
    ft_t0 = [c for c in ft_cols if not _PAST_SUFFIX_RE.search(c)]
    df_tax_f = df_tax[df_tax.index.isin(ft_t0)]

    if df_tax_f.shape[0] == 0:
        raise ValueError("Taxonomy data does not match with feature table.")

    return df_tax_f


@helper_function
def _load_phylogeny(path_to_phylo: str) -> skbio.TreeNode:
    """Load a phylogenetic tree from a Newick file (.nwk)."""
    return skbio.TreeNode.read(path_to_phylo)


@helper_function
def _process_phylogeny(phylo_tree: skbio.TreeNode, ft: pd.DataFrame) -> skbio.TreeNode:
    """Prune phylogenetic tree to match feature table columns."""
    # Reference the t0 snapshot feature set (unsuffixed columns)
    t0_cols = [c for c in ft.columns if not _PAST_SUFFIX_RE.search(c)]
    # strip the 'F' prefix to match phylotree leaf names
    tip_names = [col[1:] for col in t0_cols]

    # prune tree to keep only tips present in the feature table
    tree_phylo_f = phylo_tree.shear(tip_names)

    # add prefix "F" to leaf names in tree to remain consistent with ft
    for node in tree_phylo_f.tips():
        node.name = "F" + node.name

    # ensure that # leaves in tree == feature table dimension
    num_leaves = tree_phylo_f.count(tips=True)
    assert num_leaves == len(t0_cols)

    return tree_phylo_f


@helper_function
def _extract_mlflow_logs_to_csv(tracking_uri: str, output_dir: str) -> None:
    """Extract MLflow run logs to a consolidated CSV file.

    Reads all experiments and runs from the MLflow tracking backend and saves
    them as a single CSV with run info, parameters, metrics, and tags.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = client.search_experiments()

    exp_ids = [e.experiment_id for e in experiments]
    if not exp_ids:
        return

    exp_name_map = {e.experiment_id: e.name for e in experiments}

    # MLflow's search_runs is paginated (default max_results=1000); loop on
    # the page token to retrieve every run.
    all_runs = []
    page_token = None
    while True:
        page = client.search_runs(
            experiment_ids=exp_ids,
            page_token=page_token,
        )
        all_runs.extend(page)
        page_token = getattr(page, "token", None)
        if not page_token:
            break

    if not all_runs:
        return

    all_run_data = []
    for run in all_runs:
        row = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "experiment_name": exp_name_map.get(run.info.experiment_id, ""),
            "status": run.info.status,
            "start_time": pd.Timestamp(run.info.start_time, unit="ms")
            if run.info.start_time
            else None,
            "end_time": pd.Timestamp(run.info.end_time, unit="ms")
            if run.info.end_time
            else None,
        }
        row.update({f"params.{k}": v for k, v in run.data.params.items()})
        row.update({f"metrics.{k}": v for k, v in run.data.metrics.items()})
        row.update({f"tags.{k}": v for k, v in run.data.tags.items()})
        all_run_data.append(row)

    df = pd.DataFrame(all_run_data)
    df.to_csv(os.path.join(output_dir, "mlflow_logs.csv"), index=False)


@helper_function
def _define_model_tracker(tracking_uri: str, path_exp: str) -> str:
    if tracking_uri == "mlruns":
        db_path = os.path.join(path_exp, "mlflow.db")
        path_tracker = f"sqlite:///{db_path}"
        print(
            "MLflow tracking enabled. Logs will be extracted to CSV after "
            "experiment completion. To monitor live progress run: "
            f"mlflow ui --backend-store-uri {path_tracker}"
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

    # ! Process taxonomy and phylogeny by microbial feature table
    ft_col = [x for x in train_val.columns if x.startswith("F")]
    if tax is not None:
        tax = _process_taxonomy(tax, train_val[ft_col])
    if tree_phylo is not None:
        tree_phylo = _process_phylogeny(tree_phylo, train_val[ft_col])

    # ! Determine K for cross-validation inside each trial. When the config
    # does not specify ``k_folds`` we derive a problem-aware default from the
    # number of independent units (groups when ``group_by_column`` is set,
    # samples otherwise), capped by the smallest stratum when stratifying.
    # Setting ``k_folds: 1`` falls back to the original single-split behavior.
    k_folds = adaptive_k_folds(
        train_val,
        group_by_column=config.get("group_by_column"),
        stratify_by=config.get("stratify_by"),
        target=config.get("target"),
        task_type=config.get("task_type", "regression"),
        requested=config.get("k_folds"),
    )
    print(f"K-fold cross-validation inside trainable: k_folds={k_folds}")

    # ! Run all experiments in a temporary directory to reduce inode usage.
    # Trial directories and MLflow logs are created here, then only the
    # consolidated outputs (best models, MLflow CSV) are kept in path_exp.
    with tempfile.TemporaryDirectory() as tmp_storage:
        path_tracker = _define_model_tracker(config["tracking_uri"], tmp_storage)

        result_dic = run_all_trials(
            train_val,
            config["target"],
            config["group_by_column"],
            config.get("stratify_by", None),
            config["seed_data"],
            config["seed_model"],
            tax,
            tree_phylo,
            path_tracker,
            tmp_storage,
            # time_budget for search
            config["time_budget_s"],
            config["max_cuncurrent_trials"],
            config["experiment_tag"],
            model_types=config["ls_model_types"],
            fully_reproducible=config["fully_reproducible"],
            model_hyperparameters=config.get("model_hyperparameters", {}),
            optuna_searchspace_sampler=config.get(
                "optuna_searchspace_sampler", "TPESampler"
            ),
            task_type=config.get("task_type", "regression"),
            k_folds=k_folds,
        )

        # ! Get best models of this experiment
        best_model_dic = retrieve_n_init_best_models(result_dic, train_val)

        # ! Extract MLflow logs to CSV before temp directory cleanup
        if config["tracking_uri"] == "mlruns":
            _extract_mlflow_logs_to_csv(path_tracker, path_exp)
            print(f"MLflow logs saved to: {os.path.join(path_exp, 'mlflow_logs.csv')}")

    # Update model paths to the permanent experiment directory
    for tmodel in best_model_dic.values():
        tmodel.path = path_exp

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
        path_to_tax (str, optional): Path to taxonomy file (.tsv) with columns
        Feature ID (index), Taxon, Confidence. Feature IDs must match features
        starting with 'F' in `train_val` w/o the prefix 'F'. Needed for training
        trac models and feature engineering based on taxonomy. Defaults to None.
        path_to_tree_phylo (str, optional): Path to rooted phylogenetic tree in
        Newick format (.nwk) for features starting with "F" in `train_val`.
        Needed for training trac models. Defaults to None.
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
