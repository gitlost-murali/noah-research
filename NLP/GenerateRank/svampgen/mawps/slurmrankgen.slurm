#!/bin/bash
#SBATCH --time=0:59:59
#SBATCH --partition=gpushorrt
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH -J mathrankgen_svamp
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.m.c.kondragunta@student.rug.nl


module load Anaconda3/2022.05
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
pip install rich
pip install rankgen

python RANKgen-svamp.py
