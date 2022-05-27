#!/bin/bash
#SBATCH --time=17:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --job-name=heart_job
#SBATCH --mem=4GB
#SBATCH --partition=regular
module add Biopython/1.78-foss-2020a-Python-3.8.2
python3 recorder.py --name peregrine_var_noise --cores 6