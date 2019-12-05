#!/bin/bash

#SBATCH --job-name=multi-tf
#SBATCH --account=pi-lgrandi
#SBATCH --partition=xenon1t,dali
#SBATCH --partition=xenon1t
#SBATCH --qos=xenon1t

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=01:00:00

#SBATCH --output=log.txt
#SBATCH --error=log.txt

source ~/.bash/.setup_pax_head.sh

srun python wrapper.py
