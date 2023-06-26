#!/bin/bash
  
### Here are the SBATCH parameters that you should always consider:
#SBATCH --time=1-0:00:00    ## days-hours:minutes:seconds
#SBATCH --mem 64G         ## 3GB ram (hardware ratio is < 4GB/core)
#SBATCH --ntasks=1          ## Not strictly necessary because default is 1
#SBATCH --cpus-per-task=16   ## Use greater than 1 for parallelized jobs
#SBATCH --gres=gpu:A100:4


module load a100
module load anaconda3

source activate citgen

CALL TARINING SCRIPT HERE
