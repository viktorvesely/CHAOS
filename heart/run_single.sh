#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=hyper_optimization
#SBATCH --mem=8GB
#SBATCH --partition=short
module load Python/3.9.5-GCCcore-10.3.0
source ../../scipy18/bin/activate
python3 hospital.py --name hyper_peregrine --traincores 1