#!/bin/bash
#SBATCH --account=def-cpsmcgil
#SBATCH --gres=gpu:2              # Number of GPUs (per node)
#SBATCH --mem=12000M               # memory (per node)
#SBATCH --time=00:12:00
module purge
module load  python/3.6.3
source ~/ENV/bin/activate
python --version
python experiment.py
#sleep 36000
