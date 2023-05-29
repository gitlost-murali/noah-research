#!/bin/bash
#SBATCH --time=07:59:59
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH -J 5e4lr_finetuning
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.m.c.kondragunta@student.rug.nl


module load Anaconda3/2022.05
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0


python train_rankgen.py --batch_size 2 --lr 5e-4
