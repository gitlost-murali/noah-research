#!/bin/bash
#SBATCH --time=0:59:59
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1
#SBATCH -J mbartranking_svamp_wsplit
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.m.c.kondragunta@student.rug.nl


python multimodelt5rank.py   --dataset_name svamp \
       --train_file ../data/mawps_asdiv-a_svamp/trainset_nodup.json \
       --test_file ../data/mawps_asdiv-a_svamp/testset_nodup.json
       --output_dir multimodel_svamp_ranker/ \
       --modelt5_path ../debugmodels/svamp_t5_batch_output/generator_Mar_03_2023_svamp_infix/saved_model/ 
       --modelcodet5_path Salesforce/codet5-large-ntp-py
       --max_source_length 150     
       --max_target_length 64     
       --learning_rate 5e-5     
       --num_train_epochs 50     
       --logging_steps 100     
       --per_gpu_train_batch_size 4     
       --src_lang en_XX     
       --tgt_lang en_XX     
       --test_per_epoch 1     
       --regenerate 1    
       --rule_negatives 0     
       --num_negatives 2     
       --eqn_order infix     --data_limit 1 --no_cuda --debug_preds