#!/bin/bash

#SBATCH -A es_bokulich

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --tasks-per-node=2

#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --job-name="syn_cpu"
#SBATCH --mem-per-cpu=1024
#SBATCH --output="syn_cpu_out.txt"
#SBATCH --error="syn_cpu_err.txt"
#SBATCH --open-mode=append

set -x

echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS_PER_TASK: $SLURM_GPUS_PER_TASK"


# __doc_head_address_start__
# script was edited from:
# https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK:-0}" --block &
# __doc_head_ray_end__

# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK:-0}" --block &
    sleep 5
done
# __doc_worker_ray_end__


# __doc_script_start__
python -u q2_ritme/run_n_eval_tune.py --config q2_ritme/run_config.json
sstat -j $SLURM_JOB_ID
