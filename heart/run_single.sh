#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=hyper_optimization
#SBATCH --mem=8GB
#SBATCH --partition=short
module load Biopython/1.78-foss-2020a-Python-3.8.2
python3 hospital.py --name hyper_peregrine --traincores 1