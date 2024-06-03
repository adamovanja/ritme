#!/bin/bash

#SBATCH --job-name="r_5c_cpu_t10_compare"
#SBATCH -A partition_name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:59:59
#SBATCH --mem-per-cpu=1024
#SBATCH --output="%x_out.txt"
#SBATCH --open-mode=append

python q2_ritme/eval_best_trial_overall.py --model_path "experiments/models"
