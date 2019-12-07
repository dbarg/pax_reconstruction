#!/bin/bash

#SBATCH --job-name=nn_bigmem
#SBATCH --time=24:00:00

#SBATCH --partition=bigmem2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128GB

#SBATCH --output=log_s2waveforms_xy.txt # output log file
#SBATCH --error=log_s2waveforms_xy.txt  # error file

#SBATCH --mail-type=END
#SBATCH --mail-user=dbarge

source ~/.bash/.setup_ml.sh

echo "Starting..."

srun python ../nn_s2waveforms_xy_train.py -directory /project2/lgrandi/dbarge/pax_merge/temp_s2/ -max_dirs 2 -events_per_batch 10 -downsample 2

# Add lines here to run your GPU-based computations.
echo "Done"
