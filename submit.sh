#!/bin/bash
#SBATCH -J aes_bruteforce
#SBATCH -p gpu
#SBATCH -N 121                      # 1 master + 120 worker nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2                # at least 2 GPUs per node
#SBATCH -o ./logs/job_%j.out
#SBATCH -e ./logs/job_%j.err
#SBATCH --time=24:00:00

module load cuda mpi

mkdir -p logs

# Build (if not already done)
if [ ! -f build/aes_bruteforce ]; then
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)
fi

# Run: rank 0 = master, ranks 1..120 = workers
mpirun -n 121 ./build/aes_bruteforce \
    --pt 3243f6a8885a308d313198a2e0370734 \
    --ct 3925841d02dc09fbdc118597196a0b32 \
    --ks 00000000000000000000000000000000 \
    --ke ffffffffffffffffffffffffffffffff
