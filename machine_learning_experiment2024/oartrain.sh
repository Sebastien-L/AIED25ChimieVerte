#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=64
#SBATCH --mem=128G
#SBATCH --gpus=a100_7g.80gb:1
#SBATCH --time=12000
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load python/anaconda3 cuda/11.5
conda activate chimiegpu
nvidia-smi -L

/home/lalle/.conda/envs/chimiegpu/bin/python /home/lalle/ChimieVerte/machine_learning_experiment2024/train_model.py
