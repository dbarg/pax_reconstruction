#!/bin/bash

#SBATCH --job-name=merge
#SBATCH --account=pi-lgrandi
##SBATCH --partition=xenon1t,dali
##SBATCH --partition=xenon1t
#SBATCH --qos=xenon1t

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=24:00:00

#SBATCH --output=log_s2.txt
#SBATCH --error=log_s2.txt

source ~/.bash/.setup_pax_head.sh

./
echo "Done"
