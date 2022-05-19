#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --job-name=hyper_optimization
#SBATCH --mem=8GB
#SBATCH --partition=short
module load Python/3.9.5-GCCcore-10.3.0
python3 hospital.py --name hyper_tuning --parts 1 --hypercores 10