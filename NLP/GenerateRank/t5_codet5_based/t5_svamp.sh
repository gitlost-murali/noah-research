#!/bin/bash

python train_gen_t5_svamp.py \
    --train_file data/svamp/mawps-asdiv-a_svamp/train_split.csv \
    --valid_file data/svamp/mawps-asdiv-a_svamp/dev_split.csv \
    --test_file data/svamp/mawps-asdiv-a_svamp/dev.csv \
    --output_dir t5_output/generator \
    --model_path t5-large \
    --max_source_length 150 \
    --max_target_length 64 \
    --learning_rate 5e-5 \
    --num_train_epochs 50 \
    --logging_steps 100 \
    --per_gpu_train_batch_size 4 \
    --src_lang en_XX \
    --tgt_lang en_XX \
    --test_per_epoch 1 \
    --dataset_name svamp


python train_gen_t5_batching.py \
    --dataset_name mawps \
    --train_file data/mawps-single-five-fold/train_0.json \
    --valid_file data/mawps-single-five-fold/test_0.json \
    --test_file data/mawps-single-five-fold/test_0.json \
    --output_dir t5_datasets_debug/generator \
    --model_path t5-small \
    --max_source_length 150 \
    --max_target_length 64 \
    --learning_rate 5e-5 \
    --num_train_epochs 50 \
    --logging_steps 100 \
    --per_gpu_train_batch_size 4 \
    --src_lang en_XX \
    --tgt_lang en_XX \
    --test_per_epoch 1

python train_gen_t5_batching.py \
    --dataset_name svamp \
    --train_file ../data/mawps_asdiv-a_svamp/trainset_nodup.json \
    --valid_file ../data/mawps_asdiv-a_svamp/testset_nodup.json \
    --test_file ../data/mawps_asdiv-a_svamp/testset_nodup.json \
    --output_dir t5_datasets_debug/generator \
    --model_path t5-small \
    --max_source_length 150 \
    --max_target_length 64 \
    --learning_rate 5e-5 \
    --num_train_epochs 50 \
    --logging_steps 100 \
    --per_gpu_train_batch_size 4 \
    --src_lang en_XX \
    --tgt_lang en_XX \
    --test_per_epoch 1