#!/bin/bash

#SBATCH -J job_name

#SBATCH -p devq

#SBATCH -n 4

#SBATCH -N 4

#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:4

#SBATCH -o ./test_%j.out

#SBATCH -e ./test_%j.out
