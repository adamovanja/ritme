#!/bin/bash

#SBATCH --job-name="r_optuna_own_ss_rf"
#SBATCH -A es_bokulich
#SBATCH --nodes=1
#SBATCH --cpus-per-task=100
#SBATCH --time=119:59:59
#SBATCH --mem-per-cpu=4096
#SBATCH --output="%x_out.txt"
#SBATCH --open-mode=append

set -x

echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS_PER_TASK: $SLURM_GPUS_PER_TASK"

# ! USER SETTINGS HERE
# -> config file to use
CONFIG="q2_ritme/r_optuna_own_ss_rf.json"

# if your number of threads are limited increase as needed
ulimit -u 60000
ulimit -n 524288
# ! USER END __________

python -u q2_ritme/run_n_eval_tune.py --config $CONFIG
sstat -j $SLURM_JOB_ID

# get elapsed time of job
echo "TIME COUNTER:"
sacct -j $SLURM_JOB_ID --format=elapsed --allocations
