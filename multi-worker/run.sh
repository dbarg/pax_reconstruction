#!/bin/bash

#SBATCH --job-name=multi-tf
#SBATCH --account=pi-lgrandi
#SBATCH --partition=xenon1t,dali
#SBATCH --partition=xenon1t
#SBATCH --qos=xenon1t

#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=24:00:00

#SBATCH --output=log.txt
#SBATCH --error=log.txt

source ~/.bash/.setup_pax_head.sh

echo "Hello"
python ./multi-worker.py
echo "Done"
