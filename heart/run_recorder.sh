#!/bin/bash
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --job-name=heart_job
#SBATCH --mem=8GB
#SBATCH --partition=regular
module load Python/3.9.5-GCCcore-10.3.0
source scipy18/bin/activate
python3 recorder.py --name peregrine --cores 10