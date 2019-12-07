#!/bin/bash

#SBATCH --job-name=nn_gpu
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1GB

#SBATCH --partition=gpu2
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=12GB

#SBATCH --mail-type=END
#SBATCH --mail-user=dbarge

#SBATCH --output=log_s2waveforms_xy_ts250.txt # output log file
#SBATCH --error=log_s2waveforms_xy_ts250.txt  # error file

source ~/.bash/.setup_ml.sh

srun python ../nn_s2waveforms_xy_train.py -directory /project2/lgrandi/dbarge/pax_merge/temp_s2/ -gpu true -max_dirs 11 -events_per_batch 100 -epochs 10 -downsample 4

echo "Done"
