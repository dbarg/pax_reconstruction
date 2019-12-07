#!/bin/bash

#SBATCH --job-name=nn_cpu
#SBATCH --time=48:00:00

#SBATCH --account=pi-lgrandi
#SBATCH --partition=xenon1t,dali
#SBATCH --qos=xenon1t

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64GB

##SBATCH --partition=bigmem2
##SBATCH --mem=128GB

#SBATCH --output=log_s2waveforms_xy.txt # output log file
#SBATCH --error=log_s2waveforms_xy.txt  # error file

source ~/.bash/.setup_ml.sh

echo "Starting..."

srun python ../nn_s2waveforms_xy_train.py -directory /project2/lgrandi/dbarge/pax_merge/temp_s2/ -max_dirs 2 -events_per_batch 100 -downsample 1

# Add lines here to run your GPU-based computations.
echo "Done"
