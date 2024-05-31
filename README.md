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

## Model training locally
To train models with a defined configuration in `q2_ritme/config.json` run:
````
python q2_ritme/run_n_eval_tune.py --config q2_ritme/run_config.json
````

Once you have trained some models, you can check the progress of the trained models by launching `mlflow ui --backend-store-uri experiments/mlruns`.

To evaluate the best trial (trial < experiment) of all launched experiments, run:
````
python q2_ritme/eval_best_trial_overall.py --model_path "experiments/models"
````

## Model training on HPC with slurm:
Edit file `launch_slurm_syn_cpu.sh` and then run
````
sbatch launch_slurm_syn_cpu.sh
````
If you (or your collaborators) plan to launch multiple jobs on the same infrastructure you should set the variable `JOB_NB` in `launch_slurm_cpu.sh` accordingly. This variable makes sure that the assigned ports don't overlap (see [here](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#slurm-networking-caveats)).

If you are using SLURM and get the following error returned: "RuntimeError: can't start new thread"
it is probably caused by your hardware. Try decreasing the CPUs allocated to the job and/or decrease the variable `max_concurrent_trials` in `tune_models.py`.

## Call graphs
To create a call graph for all functions in the package, run the following commands:
````
pip install pyan3==1.1.1

pyan3 q2_ritme/**/*.py --uses --no-defines --colored --grouped --annotated --svg --exclude 'q2_ritme/evaluate_all_experiments.py' --exclude 'q2_ritme/eval_best_trial_overall.py' --exclude 'q2_ritme._version' > call_graph.svg
````
(Note: different other options to create call graphs were tested such as code2flow and snakeviz. However, these although properly maintained didn't directly output call graphs such as pyan3 did.)

## Background
### Why ray tune?
"By using tuning libraries such as Ray Tune we can try out combinations of hyperparameters. Using sophisticated search strategies, these parameters can be selected so that they are likely to lead to good results (avoiding an expensive exhaustive search). Also, trials that do not perform well can be preemptively stopped to reduce waste of computing resources. Lastly, Ray Tune also takes care of training these runs in parallel, greatly increasing search speed." [source](https://docs.ray.io/en/latest/tune/examples/tune-xgboost.html#tune-xgboost-ref)
