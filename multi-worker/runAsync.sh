#!/bin/bash

#SBATCH --job-name=multi-tf
#SBATCH --account=pi-lgrandi
#SBATCH --partition=xenon1t,dali
#SBATCH --partition=xenon1t
#SBATCH --qos=xenon1t

#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=24:00:00

#SBATCH --output=log.txt
#SBATCH --error=log.txt

source ~/.bash/.setup_pax_head.sh

echo "Starting..."

python './wrapper.py'
     
echo "Done"
