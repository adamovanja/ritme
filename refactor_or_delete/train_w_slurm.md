This is DEPRECATED and needs to be updated.

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
