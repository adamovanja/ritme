# q2-ritme
![CI](https://github.com/adamovanja/q2-ritme/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/adamovanja/q2-ritme/branch/main/graph/badge.svg?token=VQ4D7FXMCB)](https://codecov.io/gh/adamovanja/q2-ritme)

Longitudinal modeling approaches accounting for high-dimensional, sparse and compositional nature of microbial time-series.

## Why q2-ritme?
* q2-ritme allows optimized application of various feature engineering and modelling methods: usually optimal hyperparameters (e.g. regularization) depend on the feature transformation that is performed. q2-ritme can infer feature transformation and optimal model in one go.

## Setup
To install the required dependencies for this package run (note: running `conda activate` before `make dev` is a mandatory step to ensure also coral_pytorch is installed):
```shell
make create-env
conda activate ritme
make dev
```

## Model training
The model configuration is defined in `q2_ritme/run_config.json`. If you want to parallelise the training of different model types, we recommend training each model in a separate experiment. If you decide to run several model types in one experiment, be aware that the model types are trained sequentially. So, this will take longer to finish.

Once you have trained some models, you can check the progress of the trained models in the tracking software you selected (see section #model-tracking).

To define a suitable model configuration, please find the description of each variable in `q2_ritme/run_config.json` here:

| Parameter | Description |
|-----------|-------------|
| experiment_tag | Name of the experiment. |
| host_id | Column name for unique host_id in the metadata. |
| target | Column name of the target variable in the metadata. |
| ls_model_types | List of model types to explore sequentially - options include "linreg", "trac", "xgb", "nn_reg", "nn_class", "nn_corn" and "rf". |
| models_to_evaluate_separately | List of models to evaluate separately during iterative learning - only possible for "xgb", "nn_reg", "nn_class" and "nn_corn". |
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

### Local training
To locally train models with a defined configuration in `q2_ritme/run_config.json` run:
````
./launch_local.sh q2_ritme/run_config.json
````

To evaluate the best trial (trial < experiment) of all launched experiments locally, run:
````
python q2_ritme/eval_best_trial_overall.py --model_path "experiments/models"
````

### Training with slurm on HPC
To train a model with slurm on 1 node, edit the file `launch_slurm_cpu.sh` and then run
````
sbatch launch_slurm_cpu.sh
````

To train a model with slurm on multiple nodes or to enable running of multiple ray instances on the same HPC, you can use: `sbatch launch_slurm_cpu_multi_node.sh`. If you (or your collaborators) plan to launch multiple jobs on the same infrastructure you should set the variable `JOB_NB` in `launch_slurm_cpu_multi_node.sh` accordingly. This variable makes sure that the assigned ports don't overlap (see [here](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#slurm-networking-caveats)). Currently, the script allows for 3 parallel ray slurm jobs to be executed.
**Note:** training a model with slurm on multiple nodes can be very specific to your infrastructure. So you might need to adjust this bash script to your set-up.

#### Some common slurm errors:
If you are using SLURM and ...
* ... get the following error returned: "RuntimeError: can't start new thread" it is probably caused by thread limits of the cluster. You can try increasing the number of threads allowed `ulimit -u` in  `launch_slurm_cpu.sh` and/or decrease the variable `max_concurrent_trials` in `q2_ritme/config.json`. In case neither helps, it might be worth contacting the cluster administrators.

* ... your error message contains this: "The process is killed by SIGKILL by OOM killer due to high memory usage", you should increase the assigned memory per CPU (`--mem-per-cpu`) in  `launch_slurm_cpu.sh`.

## Model tracking
In the config file you can choose to track your trials with MLflow (tracking_uri=="mlruns") or with WandB (tracking_uri=="wandb").

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

## Developers topics - to be removed prior to publication
### Code test coverage
To run test coverage with Code Gutters in VScode run:
````
pytest --cov=q2_ritme q2_ritme/tests/ --cov-report=xml:coverage.xml
````

### Call graphs
To create a call graph for all functions in the package, run the following commands:
````
pip install pyan3==1.1.1

pyan3 q2_ritme/**/*.py --uses --no-defines --colored --grouped --annotated --svg --exclude 'q2_ritme/evaluate_all_experiments.py' --exclude 'q2_ritme/eval_best_trial_overall.py' --exclude 'q2_ritme._version' > call_graph.svg
````
(Note: different other options to create call graphs were tested such as code2flow and snakeviz. However, these although properly maintained didn't directly output call graphs such as pyan3 did.)

### Background
#### Why ray tune?
"By using tuning libraries such as Ray Tune we can try out combinations of hyperparameters. Using sophisticated search strategies, these parameters can be selected so that they are likely to lead to good results (avoiding an expensive exhaustive search). Also, trials that do not perform well can be preemptively stopped to reduce waste of computing resources. Lastly, Ray Tune also takes care of training these runs in parallel, greatly increasing search speed." [source](https://docs.ray.io/en/latest/tune/examples/tune-xgboost.html#tune-xgboost-ref)
