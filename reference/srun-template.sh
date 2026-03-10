srun hostname -s | sort | uniq > ./hostfile/$SLURM_JOB_ID.host
mpirun -np 4 --hostfile ./hostfile/$SLURM_JOB_ID.host