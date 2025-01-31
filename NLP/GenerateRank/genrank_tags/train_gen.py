# coding=utf-8
# 2021.03.10 - Changed for Generate & Rank multitask framework
#      Huawei Technologies Co., Ltd.
# Copyright 2022 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

import json
import random
import time

import numpy as np
import pandas as pd
import torch

import datetime
import wandb

from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from transformers import T5ForConditionalGeneration, T5ForConditionalGeneration
from transformers import (T5Tokenizer, AutoTokenizer, CONFIG_NAME,
                         WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)

import sys
sys.path.append('../')

from utils import read_json
from data_utils import extract_text_label, create_newtrainval_splits, add_tag_tosent
from t5_inference_utils import batch_test

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, tokenizer, data_file, dataset_name,
                 max_source_length, max_target_length,
                 eqn_order, data_limit, add_tag, gentag, ranktag):
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.eqn_order = eqn_order
        self.data_limit = data_limit
        self.add_tag = add_tag
        self.gentag = gentag
        self.ranktag = ranktag
        self.features = {}
        self.prepare_data()

    def store_feature(self, indx, goal, proof):
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            src_texts=goal,
            tgt_texts=proof,
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            src_lang=SRC_LANG,
            tgt_lang=TGT_LANG,
            padding="max_length"
        ).data
        for k,v in batch_encoding.items():
            batch_encoding[k] = torch.tensor(v)

        self.features[indx] = batch_encoding

    def prepare_data(self):
        if self.dataset_name == "mawps" or self.dataset_name == "svamp":
            # open a json file and read it where text is in "text" key and infix equation is in "template_equ" key
            data = read_json(self.data_file)
            for i, item in enumerate(tqdm(data, desc="Prepare train data")):
                goal, proof, numbers = extract_text_label(item, self.eqn_order)
                goal = add_tag_tosent(problem=goal, tag=self.gentag, add_tag=self.add_tag)
                self.store_feature(i, goal, proof)
                if self.data_limit > 0 and i > self.data_limit: break
        else:
            with open(self.data_file, encoding="utf-8") as f:
                lines = f.readlines()
            for i, line in enumerate(tqdm(lines, desc="Prepare train data")):
                items = line.strip().split('\t')
                goal, proof = items[0], items[1]
                goal = add_tag_tosent(problem=goal, tag=self.gentag, add_tag=self.add_tag)
                self.store_feature(i, goal, proof)
                if self.data_limit > 0 and i > self.data_limit: break

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def train(args, tokenizer, device):
    """ Train the model """

    train_dataset = TextDataset(tokenizer, args.train_file, args.dataset_name,
                                args.max_source_length, args.max_target_length,
                                args.eqn_order, args.data_limit, args.add_tag,
                                args.gentag, args.ranktag)
    if args.dataset_name == "svamp" or args.dataset_name == "mawps":
        valid_lines = read_json(args.valid_file)
        test_lines = read_json(args.test_file)
    else:
        with open(args.valid_file) as f:
            valid_lines = f.readlines()
        with open(args.test_file) as f:
            test_lines = f.readlines()
    
    if args.debug_preds:
        valid_lines = valid_lines[:args.data_limit]
        test_lines = test_lines[:args.data_limit]

    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))

    config = model.config
    model.to(device)

    # model size
    size = 0
    for n, p in model.named_parameters():
        size += p.nelement()
    logger.info('Total parameters: {}'.format(size))

    # Training
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank)
        extra_args = {'sampler': train_sampler}
    else:
        extra_args = {}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.per_gpu_train_batch_size,
                                                  drop_last=True, **extra_args)

    output_logging_file = os.path.join(args.output_dir, "log.txt")
    logger.info(f"len dataloader: {len(train_dataloader)}\n")
    logger.info(f"local rank: {args.local_rank}\n")
    logger.info(f"rank: {args.rank}\n")


    t_total = int(len(train_dataloader) * 100)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    logger.info("args.pos_weight_decay is None!")
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * t_total),
                                                num_training_steps=t_total)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                           output_device=args.local_rank)


    # Train!
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    epoch = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    best_valid_acc = best_valid_epoch = 0
    test_acc = {(k+1): 0 for k in range(args.num_seq)}

    for _ in train_iterator:
        epoch += 1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        train_start_time = time.time()
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            model.train()
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()
            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if global_step % args.logging_steps == 0 and args.rank==0:
                result = {}
                result['train_loss'] = (tr_loss - logging_loss) / args.logging_steps
                result['global_step'] = global_step
                result['epoch'] = epoch
                logging_loss = tr_loss
                with open(output_logging_file, "a") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
                    writer.write("\n")

                wandb.log(result)

        # epoch end
        train_end_time = time.time()
        if args.rank == 0 and epoch % args.test_per_epoch == 0:
            result = {}
            result['global_step'] = global_step
            result['epoch'] = epoch
            test_start_time = time.time()
            valid_acc, valid_total, valid_preds = batch_test(model, tokenizer = tokenizer, lines = valid_lines,
                                         device = device, num_return_sequences=1, # for valid, no topk needed
                                          dataset_name = args.dataset_name, eqn_order=args.eqn_order,
                                          batch_size=args.per_gpu_train_batch_size,
                                          add_tag = args.add_tag, gentag = args.gentag, ranktag = args.ranktag)
            test_end_time = time.time()
            valid_acc = valid_acc[1] / valid_total # Always use the first class as the positive class 
            result['valid_acc'] = valid_acc

            # store test_preds into a json file
            if args.debug_preds:
                debug_file = os.path.join(args.output_dir,"debug-valid-preds.json")
                with open(debug_file, "w", encoding="utf8") as fh:
                    json.dump(valid_preds, fh, indent=4)

            wandb.log(result)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_valid_epoch = epoch
                test_acc, test_total, test_preds = batch_test(model, tokenizer = tokenizer, lines = test_lines,
                                         device = device, num_return_sequences=args.num_seq,
                                          dataset_name = args.dataset_name, eqn_order=args.eqn_order,
                                          batch_size=args.per_gpu_train_batch_size,
                                          add_tag = args.add_tag, gentag = args.gentag, ranktag = args.ranktag)
                for k, acc in test_acc.items():
                    test_acc[k] = acc / test_total

                # store test_preds into a json file
                if args.debug_preds:
                    debug_file = os.path.join(args.output_dir,"debug-preds.json")
                    with open(debug_file, "w", encoding="utf8") as fh:
                        json.dump(test_preds, fh, indent=4)

                logging.info("** ** * Saving fine-tuned model ** ** * ")
                # Only save the model it-self
                output_dir = os.path.join(args.output_dir, "saved_model")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                model_to_save = model.module if hasattr(model, 'module') else model

                model_name = WEIGHTS_NAME

                output_model_file = os.path.join(output_dir, model_name)
                output_config_file = os.path.join(output_dir, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                config.to_json_file(output_config_file)
                tokenizer.save_pretrained(output_dir)

            for k, v in test_acc.items():
                result['test_acc_at_best_valid_' + str(k)] = v
            result['best_valid_acc'] = best_valid_acc
            result['best_valid_epoch'] = best_valid_epoch
            result['train_epoch_time'] = train_end_time - train_start_time
            result['test_time'] = test_end_time - test_start_time

            wandb.log(result)

            with open(output_logging_file, "a") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
                writer.write("\n")

    return global_step, tr_loss / global_step


def main(args):
    global SRC_LANG, TGT_LANG
    SRC_LANG = args.src_lang
    TGT_LANG = args.tgt_lang

    # Manually set the device ids.
    if not args.no_cuda:
        assert (torch.cuda.is_available()), "CUDA driver is not available.\
                                            Use --no_cuda to run on CPU."
        device_count = torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

        print('device_id (local_rank): %s' % args.local_rank)
        print('device_count: %s, rank: %s, world_size: %s' % (device_count, args.rank, args.world_size))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
        args.local_rank = -1

    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    args.n_gpu = torch.cuda.device_count()
    logger.info("args: {}".format(args))

    # Set seed
    set_seed(args)

    # Load pretrained tokenizer
    try:
        tokenizer = T5Tokenizer.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
    except: # For CodeT5
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)

    new_tokens = ['<SEP>',] + [f"#{i}" for i in range(30)]
    tokenizer.add_tokens(new_tokens)

    train(args, tokenizer, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--valid_file", default=None, type=str, required=False,
                        help="The input validing data file (a text file).")
    parser.add_argument("--test_file", default=None, type=str, required=True,
                        help="The input testing data file (a text file).")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--max_source_length", type=int, default=64)
    parser.add_argument("--max_target_length", type=int, default=32)

    ## Other parameters

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--test_per_epoch', type=int, default=1,
                        help="Test every X epochs.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training: local_rank")
    parser.add_argument("--src_lang", default='en_XX', type=str)
    parser.add_argument("--tgt_lang", default='en_XX', type=str)
    parser.add_argument('--distributed', default=False,
                        action='store_true',
                        help='Use distributed training.')
    parser.add_argument('--dataset_name', default="svamp", type=str,
                        help='Use svamp/math23k dataset.', required=True)
    parser.add_argument('--num_seq', type=int, default=10, required=False,
                        help='Number of sequences to generate for topk calculation')
    parser.add_argument('--eqn_order', default='prefix', type=str, required=True,
                        help='Order of equation to generate')
    parser.add_argument('--data_limit', default=-1, type=int, required=True,
                        help='How much data to use for training. -1 for all data.')
    parser.add_argument('--debug_preds', default=False, action='store_true',
                        help='Store predictions in a json file or not')
    parser.add_argument('--fold', default=-1, type=int,
                        help='Which fold to use for training. -1 for all data/SVAMP')
    parser.add_argument(
        "--split_ratio",
        default=0.85,
        type=float,
        help="Ratio of train-validation splits when validation split is not given",
    )
    parser.add_argument("--add_tag", default=False,
                        action="store_true", help= "Add a tag before the sentence generate: or rank:")
    parser.add_argument("--gentag", default="generate:", type=str,
                        help= "Tag generate: prepended for seq2seq task")
    parser.add_argument("--ranktag", default="rank:", type=str,
                        help= "Tag [rank] prepended for seq2seq task")

    args = parser.parse_args()

    if args.valid_file==None:
        valid_file_given = False
    else:
        valid_file_given = True

    project_name = f"{Path(args.model_path).name}-t5-Tags{args.add_tag}-newtrainval_{not valid_file_given}-{args.dataset_name}-n{args.data_limit}-{args.eqn_order}-src{args.max_source_length}-tgt{args.max_target_length}"

    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%b_%d_%Y")
    args.output_dir = Path(args.output_dir).parent/f"{Path(args.output_dir).stem}_{timestamp}_{args.dataset_name}_{args.eqn_order}"

    # Create output directory
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except:
        raise ValueError(
                    "Output directory ({}) already exists and is not empty.".format(
                        args.output_dir
                    ))

    if not valid_file_given:
        newtrainfile, newvalfile = create_newtrainval_splits(args, split_ratio=0.85)
        args.train_file = newtrainfile
        args.valid_file = newvalfile

    if args.fold != -1:
        wandb_dict = {"name": f"fold-{args.fold}-{timestamp}"}
    else:
        wandb_dict = {}
    wandb.init(project=project_name, entity="thesismurali-self", **wandb_dict)
    wandb.config = vars(args)
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    print(args)
    main(args)

