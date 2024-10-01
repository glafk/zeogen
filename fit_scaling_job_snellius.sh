#!/bin/bash
#Set job requirements
#SBATCH --reservation=jhs_tue2022
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=12:00:00 

#Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source $HOME/ondemand/thesis/bin/activate

python3 $HOME/ondemand/zeogen/source/fit_scaling_CDiVAE.py