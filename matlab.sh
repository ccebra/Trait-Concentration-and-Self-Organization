#!/bin/bash

#Here is a comment

#SBATCH --job-name=MyJob
#SBATCH --output=MyJob-%j.out
#SBATCH --error=MyJob-%j.err
#SBATCH --time=3:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --cpu_per_task=10
##SBATCH --mem-per-cpu=2000 #Per CPU

module load matlab
matlab "/home/ccebra/Bimatrix games evolution/Evolution_Test_moran_parallel.m" 

