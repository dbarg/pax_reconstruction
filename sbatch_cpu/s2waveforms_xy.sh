#!/bin/bash

#SBATCH --job-name=nn_cpu
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32GB

#SBATCH --output=log_s2waveforms_xy.txt # output log file
#SBATCH --error=log_s2waveforms_xy.txt  # error file

source ~/.bash/.setup_ml.sh

srun python ../nn_s2waveforms_xy_train.py -directory /project2/lgrandi/dbarge/pax_merge/temp_s2/ -max_dirs 2 -events_per_batch 10 -downsample 25

# Add lines here to run your GPU-based computations.
echo "Done"
