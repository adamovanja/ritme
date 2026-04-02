# ritme
[![DOI](https://zenodo.org/badge/601045059.svg)](https://doi.org/10.5281/zenodo.14149081)
![CI](https://github.com/adamovanja/ritme/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/adamovanja/ritme/graph/badge.svg?token=VQ4D7FXMCB)](https://codecov.io/gh/adamovanja/ritme)

An optimized framework for finding the best feature representation and model class of next generation sequencing data in relation to a target of interest.

If you use this software, please cite it using the metadata from `CITATION.cff`.

## Setup
*ritme* is available as a conda package on [anaconda.org](https://anaconda.org/adamova/ritme). To install it run the following command:

```shell
conda install -c adamova -c qiime2 -c conda-forge -c bioconda -c pytorch ritme
```

## Usage
*ritme* provides three main functions to prepare your data, find the best model configuration (feature + model class) for the specified target and evaluate the best model configuration on a test set. All of them can be run in the CLI or via the Python API. To see the arguments needed for each function run `ritme <function-name> --help` or have a look at the examples in the notebook [`experiments/ritme_example_usage.ipynb`](https://github.com/adamovanja/ritme/blob/main/experiments/ritme_example_usage.ipynb).

| *ritme* function       | Description                                                                      |
|------------------------|----------------------------------------------------------------------------------|
| split_train_test       | Preprocess your dataset and split it into train-test (with static/dynamic feature, stratification and grouping options)                         |
| find_best_model_config | Find the best model configuration (incl. feature representation and model class) |
| evaluate_tuned_models  | Evaluate the best model configuration on the complete train and a left-out test set                     |

## Preprocess your dataset and split it into train-test with `split_train_test`

`split_train_test` merges a metadata table (`md`) and a microbial feature table (`ft`) — both indexed by sample ID — and splits the result into training and test sets. It supports two modes: **static** and **dynamic** and allows for performing the splits in a grouped and/or stratified manner.

### Static mode (default)
Predict a target from a single time point of features and metadata:

```
ft_t, md_t  →  target_t
```

```python
train, test = split_train_test(md=metadata, ft=feature_table,
                               group_by_column="host_id", seed=42)
```

### Dynamic mode (temporal snapshotting)
When `time_col`, `host_col`, and `n_prev` are provided, the function creates sliding-window snapshots per host trajectory. This allows predicting a target from the current **and** previous time points:

```
ft_t-2, md_t-2, ft_t-1, md_t-1, ft_t, md_t  →  target_t (n_prev=2)
```

```python
train, test = split_train_test(md=metadata, ft=feature_table,
                               group_by_column="host_id", seed=42,
                               time_col="day", host_col="patient_id",
                               n_prev=2, missing_mode="exclude")
```

Past snapshot columns are suffixed (e.g. `F0__t-1`, `age__t-2`); current (t0) columns remain unsuffixed. Samples without a complete window of `n_prev` previous time points can either be discarded (`missing_mode="exclude"`) or retained with NaN-filled gaps (`missing_mode="nan"`).

### Grouping and stratification

To prevent data leakage, set `group_by_column` (e.g. `"host_id"`) so that all samples belonging to the same group end up in the same split. Additionally, `stratify_by` accepts a list of categorical columns (e.g. `["disease_state"]`) to preserve their distribution across the train/test split. When both are used, stratification is performed at the group level and the specified columns must be constant within each group.

### Restrictions and recommendations

- **Identical feature columns across snapshots** — all time points for a host must share the same set of microbial feature columns.
- **TRAC is incompatible with dynamic snapshotting** — TRAC requires a single compositional snapshot and phylogenetic tree; it is automatically removed from the model search when past snapshots are detected.
- **`missing_mode="nan"` restricts to XGBoost** — only XGBoost handles NaN values natively, so when NaN-filled snapshots are present, the model search is automatically restricted to XGBoost.
- **`time_col` must be numeric** — values are interpreted as ordered integers or floats; non-numeric values raise an error.

## Finding the best model configuration with `find_best_model_config`
The configuration of the optimization is defined in a `json` file. To define a suitable configuration for your use case, please find the description of each variable in `config/run_config.json` here:

| Parameter             | Description                                                                                                                                                                                                                                                                                                                                                |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| experiment_tag        | Name of the experiment. All outputs will be stored in the following folder: `<--path-store-model-logs>/experiment_tag` with `--path-store-model-logs` being directly provided as parameter input to the `find-best-model-config` method.                                                                                                                                                                                                                                                                                                                                     |
| fully_reproducible | Setting this to `false` ensures efficient and fast execution of trials with aggressive early stopping of bad trials. If set to `true` the trials are executed in a deterministic order at the cost of efficiency and potentially performance (see section [Note on reproducibility](#note-on-reproducibility)).
| group_by_column       | Column name to group train-test splits by (e.g. unique host_id) ensuring that rows with the same group value are not spread across multiple splits. If set to "null" no grouping is performed and default random train-test split is performed.                                                                                                            |
| target                | Column name of the target variable in the metadata.                                                                                                                                                                                                                                                                                                        |
| ls_model_types        | List of model types to explore sequentially - options include "linreg", "trac", "xgb", "nn_reg", "nn_class", "nn_corn" and "rf".                                                                                                                                                                                                                           |
| time_budget_s            | Time budget in seconds to use per model type: the larger this value the more space of the complete search space can be searched.                                                                                                                                                                                                                           |
| max_cuncurrent_trials | Maximal number of concurrent trials to run.                                                                                                                                                                                                                                                                                                                |
| seed_data             | Seed for data-related random operations.                                                                                                                                                                                                                                                                                                                   |
| seed_model            | Seed for model-related random operations.                                                                                                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                      |
| tracking_uri          | Which platform to use for experiment tracking either "wandb" for WandB or "mlruns" for MLflow. See  [model tracking](#model-tracking) for set-up instructions and [model tracking evaluation](#model-tracking-evaluation) for tips on how to evaluate the training procedure in each platform.                                                                                                                                                                                             |
| stratify_by           | Optional: List of column names to use for stratified splitting. When grouping is used, stratification is performed at the group level and the specified columns must be constant within each group. Example: `["disease_state"]`.                                                                                                                                                                                       |
| model_hyperparameters | Optional: For each model type the range of model and feature engineering hyperparameters to consider can be defined here. Additionally, for each model type the start hyperparameters to try first can be defined. Note: in case this key is not provided, the default ranges are used as defined in `model_space/static_searchspace.py`. You can find an example of a configuration file with all hyperparameters defined in `ritme/config/run_config_whparams.json` and the start hyperparameters defined for the "linreg" model. |

If you want to parallelize the training of different model types, we recommend training each model in a separate experiment [1]. If you decide to run several model types in one experiment, be aware that the model types are trained sequentially. So, this will take longer to finish.

Once you have trained some models, you can check the progress of the trained models in the tracking software you selected (see sections on [model tracking](#model-tracking) and [model training evaluation](#model-training-evaluation)).

[1] Funfact: One experiment consists of multiple trials.

## Evaluate the best model configuration with `evaluate_tuned_models`

`evaluate_tuned_models` takes the best trained models per model type from `find_best_model_config` and evaluates them on both the full training set and a held-out test set. It returns a DataFrame with performance metrics (RMSE, R², Pearson correlation) per model type and split, along with a figure of true vs. predicted scatter plots.

To prevent data leakage, all feature engineering parameters (feature selection, ALR denominator, enrichment schema, column ordering) are learned exclusively on the training set and reused as-is when predicting on held-out test data.

```python
metrics_df, fig = evaluate_tuned_models(best_model_dict, exp_config, train, test)
```

Via the CLI, metrics are saved to `best_metrics.csv` and plots to `best_true_vs_pred.png` in the experiment directory.

## Model tracking
In the run configuration file you can choose to track your trials with MLflow (`tracking_uri=="mlruns"`) or with WandB (`tracking_uri=="wandb"`).

### Choice between WandB & MLflow
To choose which tracking set-up works best for you, it is best to review the respective services: [WandB](https://docs.wandb.ai/) & [MLflow](https://mlflow.org/). In our experience, when working on a HPC cluster with limited outgoing network traffic, MLflow works better than WandB.

Independent of your choice, *ritme* is set up such that no sample-specific information is stored remotely. Any sample-specific information is stored only on your local machine. As for aggregate metrics, WandB stores these on their servers while MLflow stores them locally.

### Set-up WandB with *ritme*
In case of using WandB you need to store your `WANDB_API_KEY` & `WANDB_ENTITY` as a environment variable in `.env`. Make sure to ignore this file in version control (add to `.gitignore`)!

The `WANDB_ENTITY` is the project name you would like to store the results in. For more information on this parameter see the official webpage for initializing WandB [here](https://docs.wandb.ai/ref/python/init).

Also if you are running WandB from a HPC, you might need to set the proxy URLs to your respective URLs by exporting these variables:
```
export HTTPS_PROXY=http://proxy.example.com:8080
export HTTP_PROXY=http://proxy.example.com:8080
```
For a template on how to evaluate your models see the section on [model training evaluation](#model-training-evaluation).

### Set-up MLflow with *ritme*
In case of using MLflow you can view your models with `mlflow ui` from within the path where the logs were saved (which is outputted when running `find_best_model_config` as "You can view the model logs by launching MLflow UI from within folder : <folder_name>"). This is rather slow when many trials or experiments were launched - then viewing logs via the Python API is better suited. For more information check out the [official MLflow documentation](https://mlflow.org/docs/latest/index.html).

For a template on how to evaluate your models see the section on [model training evaluation](#model-training-evaluation).

## Model training evaluation
We provide example templates to help you evaluate your *ritme* models for both supported tracking services:
* for WandB visit [this report](https://wandb.ai/ritme/trials_wandb/reports/Template-for-ritme-training-evaluation--VmlldzoxMzE1MTQ5MQ?accessToken=2yuzgiu4ke2r3ky5c894nnygguse8xh9mt5ky3g7p43mcirbmhv504ruipny54l5) - simply copy the template and update the run set at the end of the report to your own experiment.

* for MLflow see the notebook [`experiments/evaluate_trials_mlflow.ipynb`](https://github.com/adamovanja/ritme/blob/main/experiments/evaluate_trials_mlflow.ipynb).

## Note on reproducibility
When you enable `"fully_reproducible": true` in your experiment configuration, all runs on identical hardware will produce fully reproducible results, albeit with a potential impact on efficiency and performance. This guarantee becomes particularly relevant when executing a large number of trials in parallel. (For small-scale experiments — e.g. with 2 trials — you will often observe identical results even with `"fully_reproducible": false`.)

## Contact
In case of questions or comments feel free to raise an issue in this repository.

## License
If you use this software, please cite it using the metadata from `CITATION.cff`.

*ritme* is released under a BSD-3-Clause license. See LICENSE for more details.
