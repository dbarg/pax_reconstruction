#!/bin/bash

#SBATCH --job-name=nn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=01:00:00
#SBATCH --account=pi-lgrandi
#SBATCH --partition=xenon1t,dali
#SBATCH --qos=xenon1t

#SBATCH --output=log.txt
#SBATCH --error=log.txt

#cd ${HOME}/dali/pax_reconstruction/nn_s2only
#source ~/.bash/.setup_pax_head.sh

#python ./nn_s2_train.py -directory ../../xe1t-processing/pax_merge/temp_s2/ -max_dirs 2 -events_per_batch 100 -samples 1000 -downsample 10
echo "HEELO"
