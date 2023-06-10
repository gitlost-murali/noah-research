#!/bin/bash
#SBATCH --time=01:58:59
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH -J svamp_7b_5shot
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.m.c.kondragunta@student.rug.nl

module load Anaconda3/2022.05

conda activate torchenv
pip install transformers einops accelerate xformers

pip install sentencepiece

export HUGGINGFACE_HUB_CACHE=/scratch/$USER/huggingface_hub_cache
export HF_HOME=/scratch/$USER/hf_home

python svamp_zeroshot.py --model_path /scratch/s5397294/falcon-7b/ \
                         --trainfile ../svampgen/svamp/traint5_preds.json \
                         --testfile ../svampgen/svamp/traint5_preds.json \
                         --temperature 0.1 \
                         --num_samples 5
