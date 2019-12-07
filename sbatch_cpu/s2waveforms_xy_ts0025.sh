#!/bin/bash

#SBATCH --job-name=nn_cpu
#SBATCH --time=24:00:00

#SBATCH --account=pi-lgrandi
#SBATCH --partition=xenon1t,dali
#SBATCH --qos=xenon1t

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=500MB

#SBATCH --output=log_s2waveforms_xy_ts25.txt # output log file
#SBATCH --error=log_s2waveforms_xy_ts25.txt  # error file

source ~/.bash/.setup_ml.sh

srun python ../nn_s2waveforms_xy_train.py -directory /project2/lgrandi/dbarge/pax_merge/temp_s2/ -max_dirs 22 -events_per_batch 100 -epochs 10 -downsample 40

echo "Done"
