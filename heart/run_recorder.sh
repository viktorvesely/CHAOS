#!/bin/bash
#SBATCH --time=1-16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --job-name=heart_job
#SBATCH --mem=8GB
#SBATCH --partition=regular
module add Biopython/1.78-foss-2020a-Python-3.8.2
python3 recorder.py --name peregrine_var_noise --cores 5