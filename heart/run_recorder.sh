#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --job-name=heart_job
#SBATCH --mem=2GB
#SBATCH --partition=short
module add Biopython/1.78-foss-2020a-Python-3.8.2
python3 recorder.py --name per_test --cores 2