# ritme
[![DOI](https://zenodo.org/badge/601045059.svg)](https://doi.org/10.5281/zenodo.14149081)
![CI](https://github.com/adamovanja/ritme/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/adamovanja/ritme/graph/badge.svg?token=VQ4D7FXMCB)](https://codecov.io/gh/adamovanja/ritme)

An optimized framework for finding the best feature representation and model class of next generation sequencing data in relation to a target of interest.


## Setup
To install the required dependencies for this package clone this repos and run (note: running `make dev` is a mandatory step to ensure all required packages are installed):
```shell
make create-env
conda activate ritme
make dev
```
Funfact: A properly packaged version of the package is coming soon.

## Usage
*ritme* provides three main functions to prepare your data, find the best model configuration (feature + model class) for the specified target and evaluate the best model configuration on a left-out test set. All of them can be run in the CLI or via the Python API. For examples on how to use each function see this notebook `experiments/ritme_example_usage.ipynb`:
| Method                   | Description                                                                      |
|--------------------------|----------------------------------------------------------------------------------|
| split_train_test         | Split the dataset into train-test in a stratified manner                         |
| find_best_model_config   | Find the best model configuration (incl. feature representation and model class) |
| evaluate_tuned_models.py | Evaluate the best model configuration on a left-out test set                     |

## Finding the best model configuration
The set-up of the optimization is defined in `ritme/run_config.json`. If you want to parallelize the training of different model types, we recommend training each model in a separate experiment. If you decide to run several model types in one experiment, be aware that the model types are trained sequentially. So, this will take longer to finish.

Once you have trained some models, you can check the progress of the trained models in the tracking software you selected (see section #model-tracking).

To define a suitable run configuration, please find the description of each variable in `ritme/config/run_config.json` here:

| Parameter | Description |
|-----------|-------------|
| experiment_tag | Name of the experiment. |
| stratify_by_column | Column name to stratify splits by (e.g. unique host_id). |
| target | Column name of the target variable in the metadata. |
| feature_prefix | Prefix of features variables. |
| ls_model_types | List of model types to explore sequentially - options include "linreg", "trac", "xgb", "nn_reg", "nn_class", "nn_corn" and "rf". |
| num_trials | Total number of trials to try per model type: the larger this value the more space of the complete search space can be searched. |
| max_cuncurrent_trials | Maximal number of concurrent trials to run. |
| path_to_ft | Path to the feature table file. |
| path_to_md | Path to the metadata file. |
| path_to_phylo | Path to the phylogenetic tree file. |
| path_to_tax | Path to the taxonomy file. |
| seed_data | Seed for data-related random operations. |
| seed_model | Seed for model-related random operations. |
| test_mode | Boolean flag to indicate if running in test mode. |
| tracking_uri | Which platform to use for experiment tracking either "wandb" for WandB or "mlruns" for MLflow. See  #model-tracking for set-up instructions. |
| train_size | Fraction of data to use for training (e.g., 0.8 for 80% train, 20% test split). |
| model_hyperparameters | Optional: For each model type the range of hyperparameters to check can be defined here. Note: in case this key is not provided, the default ranges are used as defined in `model_space/static_searchspace.py`. You can find an example of a configuration file with all hyperparameters defined as per default in `run_config_whparams.json`|


## Model tracking
In the run configuration file you can choose to track your trials with MLflow (tracking_uri=="mlruns") or with WandB (tracking_uri=="wandb").

### Choice between MLflow & WandB
WandB stores aggregate metrics on their servers. The way *ritme* is set up no sample-specific information is stored remotely. This information is stored on your local machine.
To choose which tracking set-up works best for you, it is best to review the respective services.

### MLflow
In case of using MLflow you can view your models with `mlflow ui --backend-store-uri experiments/mlruns`. For more information check out the [official MLflow documentation](https://mlflow.org/docs/latest/index.html).

### WandB
In case of using WandB you need to store your `WANDB_API_KEY` & `WANDB_ENTITY` as a environment variable in `.env`. Make sure to ignore this file in version control (add to `.gitignore`)!

The `WANDB_ENTITY` is the project name you would like to store the results in. For more information on this parameter see the official webpage for initializing WandB [here](https://docs.wandb.ai/ref/python/init).

Also if you are running WandB from a HPC, you might need to set the proxy URLs to your respective URLs by exporting these variables:
```
export HTTPS_PROXY=http://proxy.example.com:8080
export HTTP_PROXY=http://proxy.example.com:8080
````


## Contact
In case of questions or comments feel free to raise an issue in this repository.

## License
*ritme* is released under a BSD-3-Clause license. See LICENSE for more details.
