#!/bin/bash

#SBATCH --job-name=test
#SBATCH --time=01:00:00
#SBATCH --account=pi-lgrandi
#SBATCH --partition=xenon1t,dali
#SBATCH --qos=xenon1t

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB

#SBATCH --output=log.txt # output log file
#SBATCH --error=log.txt  # error file

source ~/.bash/.setup_ml.sh

export TMPDIR=/scratch/midway2/$USER/slurm/$SLURM_JOBID
export SLURM_TMPDIR=/scratch/midway2/$USER/slurm/$SLURM_JOBID

echo "Starting..."

srun --preserve-env --profile=All --acctg-freq=1 --task-prolog=./prolog.sh --task-epilog=/dali/lgrandi/dbarge/pax_reconstruction/profiling/epilog.sh python test.py

echo "Done"
