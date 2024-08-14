#!/bin/bash

#SBATCH --job-name="run_config"
#SBATCH -A partition_name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=100
#SBATCH --time=23:59:59
#SBATCH --mem-per-cpu=4096
#SBATCH --output="%x_out.txt"
#SBATCH --open-mode=append

set -x

echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS_PER_TASK: $SLURM_GPUS_PER_TASK"

# ! USER SETTINGS HERE
# -> config file to use
CONFIG="q2_ritme/run_config.json"
# -> count of this concurrent job launched on same infrastructure
# -> only these values are allowed: 1, 2, 3 - since below ports are
# -> otherwise taken or not allowed
JOB_NB=2

# if your number of threads are limited increase as needed
ulimit -u 60000
ulimit -n 524288
# ! USER END __________

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
port=$((6378 + JOB_NB))
node_manager_port=$((6600 + JOB_NB * 100))
object_manager_port=$((6601 + JOB_NB * 100))
ray_client_server_port=$((1 + JOB_NB * 10000))
redis_shard_ports=$((6602 + JOB_NB * 100))
min_worker_port=$((2 + JOB_NB * 10000))
max_worker_port=$((9999 + JOB_NB * 10000))
dashboard_port=$((8265 + JOB_NB))

ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" \
    --port=$port \
    --node-manager-port=$node_manager_port \
    --object-manager-port=$object_manager_port \
    --ray-client-server-port=$ray_client_server_port \
    --redis-shard-ports=$redis_shard_ports \
    --min-worker-port=$min_worker_port \
    --max-worker-port=$max_worker_port \
    --dashboard-port=$dashboard_port \
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

# Output the dashboard URL
dashboard_url="http://${head_node_ip}:${dashboard_port}"
export RAY_DASHBOARD_URL="$dashboard_url"
echo "Ray Dashboard URL: $RAY_DASHBOARD_URL"

# __doc_script_start__
python -u q2_ritme/run_n_eval_tune.py --config $CONFIG
sstat -j $SLURM_JOB_ID

# get elapsed time of job
echo "TIME COUNTER:"
sacct -j $SLURM_JOB_ID --format=elapsed --allocations
