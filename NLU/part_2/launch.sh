#!/bin/bash

#SBATCH --partition=edu-20h
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00

#SBATCH --job-name=baseline
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --account=riccardi.edu


echo "loading CUDA"
module load cuda

echo "loading Conda"
source /home/ettore.saggiorato/miniconda3/bin/activate
conda activate NLU

echo "Training"
cd /home/ettore.saggiorato/nlu/NLU/part_2/
python3 main.py

echo "Done"
