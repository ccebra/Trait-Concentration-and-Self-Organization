#!/bin/bash

#SBATCH --job-name=simpleML
#SBATCH --output=matlab_simple.out
#SBATCH --error=matlab_simple.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

module load matlab

matlab -nodisplay < matlab_simple.m