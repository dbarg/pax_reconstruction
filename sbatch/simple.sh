#!/bin/bash

#SBATCH --job-name=nn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=8:00:00
#SBATCH --account=pi-lgrandi
#SBATCH --partition=xenon1t,dali
#SBATCH --qos=xenon1t

#SBATCH --output=log.txt
#SBATCH --error=log.txt


echo "HERE"
