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
import csv
import datetime
import logging
import os
import random
import sys
import time
from pathlib import Path
import json
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from transformers import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    AdamW,
    T5Config,
    T5Tokenizer,
    AutoTokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from transformers.models.mbart.modeling_mbart import shift_tokens_right

import wandb

import sys
sys.path.append("..")

sys.path.append("../t5_codet5_based/")

from data_utils import extract_text_label, remove_invalid_equations,\
                       create_newtrainval_splits, add_tag_tosent, add_rank_eqns
from exp_tree import corrupt_expression
from t5_inference_utils import GeneralDataset
from utils import clean_text, is_equal, read_json

NUM_STR = [str(i) for i in range(10)] + ["#"]
csv.field_size_limit(sys.maxsize)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class LineDataset(Dataset):
    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        prob, label = self.lines[item].strip().split("\t")
        return {"prob": prob, "label": label}


def add_rule_negatives(lines, add_num=10, all_expressions=None):
    all_lines = []
    prev_prob = ""
    for line in lines:
        correct = line["correct_answer"]
        prob = line["prob"]
        for _ in range(add_num):
            try:
                rule_neg = corrupt_expression(correct)
                this_label = (
                    "1" if is_equal(correct, rule_neg, number_filler=True) else "0"
                )
                line["equations"].append(rule_neg)
                line["gt"].append(this_label)
            except:
                print("error??")
                print(correct)

        all_lines.append(line)
    return all_lines


class TextDataset(Dataset):
    def __init__(self, tokenizer, lines, max_source_length, max_target_length,
                 add_tag, gentag, ranktag):
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.add_tag = add_tag
        self.gentag = gentag
        self.ranktag = ranktag
        self.lines = lines
        self.tokenizer = tokenizer
        self.features = {}
        self.prepare_data()

    def prepare_data(self):
        lines = self.lines
        for i, item in enumerate(tqdm(lines, desc="Prepare train data")):
            prob, gen, correct, gt, cls_label = item["prob"], item["equations"], item["correct_answer"], item["gt"], item["cls_label"]
            # problem \t generated expression \t groundtruth \t rank_label
            # gen = [eq1, eq2, eq3, eq4, eq5]
            # gt = [1,0,0,0,1]

            if cls_label == "-1":  # prepare data for generate and revise
                # assert gen == "<mask>"
                encoder_input = prob
                encoder_input = add_tag_tosent(encoder_input, self.gentag, self.add_tag)
                decoder_input = correct
                batch_encoding = self.tokenizer.prepare_seq2seq_batch(
                    src_texts=encoder_input,
                    tgt_texts=decoder_input,
                    max_length=self.max_source_length,
                    max_target_length=self.max_target_length,
                    src_lang=SRC_LANG,
                    tgt_lang=TGT_LANG,
                    padding="max_length",
                    return_tensors="pt",
                )

                for k, v in batch_encoding.items():
                    batch_encoding[k] = v.squeeze(0)
            else:  # prepare data for rank function
                encoder_input = prob
                encoder_input = add_tag_tosent(encoder_input, self.ranktag, self.add_tag)
                encoder_input = add_rank_eqns(encoder_input, gen, self.add_tag)
                correct_eqn = [eq for eq, clslabel in zip(gen, gt) if clslabel == 1]
                try:
                    decoder_input = correct_eqn[0]
                except:
                    print("correct_eqn is empty")
                    decoder_input = correct
                batch_encoding = self.tokenizer.prepare_seq2seq_batch(
                    src_texts=encoder_input,
                    tgt_texts=decoder_input,
                    max_length=self.max_source_length,
                    max_target_length=self.max_target_length,
                    src_lang=SRC_LANG,
                    tgt_lang=TGT_LANG,
                    padding="max_length",
                    return_tensors="pt",
                )

                for k, v in batch_encoding.items():
                    batch_encoding[k] = v.squeeze(0)

            self.features[i] = batch_encoding


    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def collect_data_to_trainfile(args):
    samples = []
    for i in range(args.world_size):
        output_samples_file = os.path.join(args.output_dir, f"gen_samples_{i}.train")
        with open(output_samples_file, "r") as f:
            samples = json.load(f)
    print("writing in collect")
    with open(args.collect_file, "w") as f:
        json.dump(samples, f, indent=4)
    print("write done")


def train(args, tokenizer, device):
    """Train the model"""
    if args.dataset_name == "mawps" or args.dataset_name == "svamp":
        # open a json file and read it where text is in "text" key and infix equation is in "template_equ" key
        data = read_json(args.train_file)
        lines = []
        labels = []
        numbers_list = []
        for i, item in enumerate(tqdm(data, desc="Prepare train data")):
            goal, proof, numbers = extract_text_label(item, args.eqn_order)
            lines.append(goal)
            labels.append(proof)
            numbers_list.append(numbers)
            if args.data_limit > 0 and i > args.data_limit:
                break
        raw_train_dataset = GeneralDataset(
            prob=lines, label=labels, numbers=numbers_list
        )
    else:
        with open(args.train_file) as f:
            raw_train_lines = f.readlines()
        raw_train_dataset = LineDataset(raw_train_lines)

    if args.distributed:
        raw_train_sampler = torch.utils.data.distributed.DistributedSampler(
            raw_train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
        extra_args = {"sampler": raw_train_sampler}
    else:
        extra_args = {}
    raw_train_dataloader = torch.utils.data.DataLoader(
        raw_train_dataset,
        batch_size=args.per_gpu_train_batch_size,
        drop_last=False,
        **extra_args,
    )

    valid_lines = read_json(args.valid_file)
    test_lines = read_json(args.test_file)

    if args.debug_preds:
        valid_lines = valid_lines[: args.data_limit]
        test_lines = test_lines[: args.data_limit]

    print(f"load model from {args.model_path}")
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))
    print("model load done")

    config = model.config
    model.to(device)

    # model size
    size = 0
    for n, p in model.named_parameters():
        size += p.nelement()
    logger.info("Total parameters: {}".format(size))

    # Training
    # generate samples for revise
    output_samples_file = os.path.join(
        args.output_dir, f"gen_samples_{args.local_rank}.train"
    )
    args.collect_file = os.path.join(args.output_dir, args.collect_file)
    print("start generate samples")
    gen_start_time = time.time()
    generate_sample(
        model, tokenizer, raw_train_dataloader, device, output_samples_file, args
    )
    print(f"finish generate samples in {time.time() - gen_start_time}")

    if args.distributed:
        torch.distributed.barrier()

    if args.local_rank == 0:
        print("collecting train data")
        collect_data_to_trainfile(args)
    if args.distributed:
        torch.distributed.barrier()
    with open(args.collect_file) as f:
        collect_lines = json.load(f)
    all_expressions = None
    if args.rule_negatives:
        collect_lines = add_rule_negatives(
            collect_lines, add_num=args.num_negatives, all_expressions=all_expressions
        )
    train_dataset = TextDataset(
        tokenizer, collect_lines, args.max_source_length, args.max_target_length,
        args.add_tag, args.gentag, args.ranktag
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank
        )
        extra2_args = {"sampler": train_sampler}
    else:
        extra2_args = {}
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_gpu_train_batch_size,
        drop_last=False,
        **extra2_args,
    )

    output_logging_file = os.path.join(args.output_dir, "log.txt")
    logger.info(f"len dataloader: {len(train_dataloader)}\n")
    logger.info(f"local rank: {args.local_rank}\n")
    logger.info(f"rank: {args.rank}\n")

    t_total = int(len(train_dataloader) * args.num_train_epochs)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    logger.info("args.pos_weight_decay is None!")
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * t_total),
        num_training_steps=t_total,
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # Train!
    global_step = 0
    tr_loss = 0.0

    epoch = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    best_valid_acc = best_valid_epoch = 0
    valid_rank_acc = (
        valid_acc
    ) = test_rank_acc = test_acc = test_acc_all = valid_acc_all = -1
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

            if global_step % args.logging_steps == 0 and args.rank == 0:
                result = {}
                result["train_loss"] = tr_loss / args.logging_steps
                result["global_step"] = global_step
                result["epoch"] = epoch
                tr_loss = 0.0
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
            result["global_step"] = global_step
            result["epoch"] = epoch
            test_start_time = time.time()
            test_output_dir = os.path.join(
                args.output_dir, "{}-{}".format("test_output", global_step)
            )
            if not os.path.exists(test_output_dir):
                os.makedirs(test_output_dir)
            valid_output_file = os.path.join(test_output_dir, "output_gen.valid")

            valid_acc, valid_acc_all, valid_total = gen_test(
                model,
                device,
                tokenizer,
                valid_lines,
                args.eqn_order,
                test_file=valid_output_file,
            )

            valid_acc = valid_acc / valid_total
            valid_acc_all = valid_acc_all / valid_total

            with open(valid_output_file, "r") as f:
                valid_rank_lines = json.load(f)

            if args.remove_invalid_eqns_manually:
                valid_rank_lines = remove_invalid_equations(valid_rank_lines)
            valid_rank_acc = genrank_test(
                args, model, device, tokenizer, valid_rank_lines, tokenizer.pad_token_id
            )
            test_end_time = time.time()
            result["valid_acc"] = valid_acc
            result["valid_acc_all"] = valid_acc_all
            result["valid_rank_acc"] = valid_rank_acc
            if valid_rank_acc > best_valid_acc:
                best_valid_acc = valid_acc if valid_rank_acc == -1 else valid_rank_acc
                best_valid_epoch = epoch
                test_output_dir = os.path.join(
                    args.output_dir, "{}-{}".format("test_output", global_step)
                )
                if not os.path.exists(test_output_dir):
                    os.makedirs(test_output_dir)
                test_output_file = os.path.join(test_output_dir, "output_gen.test")

                test_acc, test_acc_all, test_total = gen_test(
                    model,
                    device,
                    tokenizer,
                    test_lines,
                    eqn_order=args.eqn_order,
                    test_file=test_output_file,
                )
                test_acc = test_acc / test_total
                test_acc_all = test_acc_all / test_total

                with open(test_output_file, "r") as f:
                    test_rank_lines = json.load(f)

                if args.remove_invalid_eqns_manually:
                    test_rank_lines = remove_invalid_equations(test_rank_lines)
                test_rank_acc = genrank_test(
                    args,
                    model,
                    device,
                    tokenizer,
                    test_rank_lines,
                    tokenizer.pad_token_id,
                )

                logging.info("** ** * Saving fine-tuned model ** ** * ")
                # Only save the model it-self
                output_dir = os.path.join(args.output_dir, "saved_model")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                model_to_save = model.module if hasattr(model, "module") else model

                model_name = WEIGHTS_NAME

                output_model_file = os.path.join(output_dir, model_name)
                output_config_file = os.path.join(output_dir, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                config.to_json_file(output_config_file)
                tokenizer.save_pretrained(output_dir)

            result["test_acc_at_best_valid"] = test_acc
            result["test_acc_all_at_best_valid"] = test_acc_all
            result["test_rank_acc_at_best_valid"] = test_rank_acc
            result["best_valid_acc"] = best_valid_acc
            result["best_valid_epoch"] = best_valid_epoch
            result["train_epoch_time"] = train_end_time - train_start_time
            result["test_time"] = test_end_time - test_start_time

            with open(output_logging_file, "a") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
                writer.write("\n")

            wandb.log(result)

        if args.distributed:
            torch.distributed.barrier()
        # generate samples for revise
        if args.regenerate == 1:
            output_samples_file = os.path.join(
                args.output_dir, f"gen_samples_{args.local_rank}.train"
            )
            generate_sample(
                model,
                tokenizer,
                raw_train_dataloader,
                device,
                output_samples_file,
                args,
            )

            if args.local_rank == 0:
                print("collecting train data")
                collect_data_to_trainfile(args)
            if args.distributed:
                torch.distributed.barrier()
            with open(args.collect_file, "r") as f:
                collect_lines = json.load(f)
            if args.rule_negatives:
                collect_lines = add_rule_negatives(
                    collect_lines,
                    add_num=args.num_negatives,
                    all_expressions=all_expressions,
                )
            if args.remove_invalid_eqns_manually:
                collect_lines = remove_invalid_equations(collect_lines)
            train_dataset = TextDataset(
                tokenizer, collect_lines, args.max_source_length, args.max_target_length,
                args.add_tag, args.gentag, args.ranktag
            )
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, num_replicas=args.world_size, rank=args.rank
                )
                extra3_args = {"sampler": train_sampler}
            else:
                extra3_args = {}
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.per_gpu_train_batch_size,
                drop_last=False,
                **extra3_args,
            )

    return global_step, tr_loss / global_step


def generate_sample(model, tokenizer, dataloader, device, output_samples_file, args):
    """
    generate negative samples using the model for revise training
    """
    samples = []
    samples_list = []
    if args.use_generate_sample:
        model = model.module if hasattr(model, "module") else model
        model.eval()
        beam_size = args.num_negatives
        for data in tqdm(dataloader):
            prob, label = data["prob"], data["label"]
            gen_prob = prob
            gen_prob = [add_tag_tosent(problem = mwp, tag = args.gentag, add_tag=args.add_tag) for mwp in gen_prob]
            batch = tokenizer.prepare_seq2seq_batch(gen_prob, return_tensors="pt")
            for k, v in batch.items():
                batch[k] = v.to(device)

            text = model.generate(
                **batch,
                num_beams=beam_size,
                early_stopping=True,
                max_length=args.max_target_length,
                num_return_sequences=beam_size,
            )  # batch * 10, len
            text = tokenizer.batch_decode(text, skip_special_tokens=True)
            text = [clean_text(t) for t in text]

            label = [clean_text(t) for t in label]

            idx = 0

            for p, e in zip(prob, label):
                local_samples_list = dict()
                local_samples_list["prob"] = p
                local_samples_list["equations"] = []
                local_samples_list["gt"] = []
                local_samples_list["correct_answer"] = e

                samples.append((p, "<mask>", e, 0))
                samples.append((p, e, e, 1))
                beam = text[idx * beam_size : (idx + 1) * beam_size]
                for b in beam:
                    if is_equal(e, b, number_filler=True):
                        samples.append((p, b, b, 1))
                        local_samples_list["equations"].append(b)
                        local_samples_list["gt"].append(1)
                    else:
                        samples.append((p, b, e, 0))
                        local_samples_list["equations"].append(b)
                        local_samples_list["gt"].append(0)
                
                local_samples_list["cls_label"] = 1
                samples_list.append(local_samples_list)
                local_samples_list["cls_label"] = -1
                samples_list.append(local_samples_list)
                idx += 1
    else:
        for data in tqdm(dataloader):
            prob, label = data["prob"], data["label"]
            prob = [add_tag_tosent(problem = mwp, tag = args.gentag, add_tag=args.add_tag) for mwp in prob]
            label = [clean_text(t) for t in label]
            for p, e in zip(prob, label):
                local_samples_list = dict()
                local_samples_list["prob"] = p
                local_samples_list["equations"] = []
                local_samples_list["gt"] = []
                local_samples_list["correct_answer"] = e

                local_samples_list["equations"].append("<mask>")
                local_samples_list["gt"].append(0)

                
                local_samples_list["equations"].append(e)
                local_samples_list["gt"].append(1)
                samples.append((p, "<mask>", e, 0))
                samples.append((p, e, e, 1))

                local_samples_list["cls_label"] = -1
                samples_list.append(local_samples_list)

                local_samples_list["cls_label"] = 1
                samples_list.append(local_samples_list)

    with open(output_samples_file, "w") as f:
        for sample in samples:
            if sample[1] != "<mask>":
                f.write(
                    f"{sample[0]}\t{sample[1]}\t{sample[2]}\t{sample[3]}\n"
                )  # for ranker
            else:
                f.write(
                    f"{sample[0]}\t{sample[1]}\t{sample[2]}\t-1\n"
                )  # for generation

    with open(output_samples_file, "w") as f:
        json.dump(samples_list, f, indent=4)

    return samples_list


def genrank_test(args, model, device, tokenizer, lines, pad_token_id, num_beam=10):
    print("rank testing..........")
    model = model.module if hasattr(model, "module") else model
    model.eval()
    # group generated expressions as batches for each problem
    prob_prev = ""
    acc = 0
    for i, item in enumerate(lines):
        prob, equations, gts, label = item["prob"], item["equations"], \
                                      item["gt"], item["correct_answer"]

        encoder_input = add_tag_tosent(prob, args.ranktag, args.add_tag)
        encoder_input = add_rank_eqns(encoder_input, equations, args.add_tag)
        correct_eqn = [eq for eq, clslabel in zip(equations, gts) if clslabel == 1]

        # add cur batch_encoding to batch
        batch = tokenizer.prepare_seq2seq_batch(
            src_texts=prob,
            max_length=args.max_source_length,
            max_target_length=args.max_target_length,
            src_lang=args.src_lang,
            padding="max_length",
            return_tensors="pt",
        )

        for k, v in batch.items():
            batch[k] = v.to(device)

        text = model.generate(
            **batch,
            num_beams=num_beam,
            early_stopping=True,
            max_length=100,
            num_return_sequences=1, # get only the best sequence
        )

        text = tokenizer.batch_decode(text, skip_special_tokens=True)
        text = [clean_text(t) for t in text]

        for idx, candidate in enumerate(text):
            if is_equal(label, candidate, number_filler=True):
                hit_acc = True
                if idx == 0:
                    hit_1 = True
                    acc += 1                


    return acc * 100.0 / len(lines)


def gen_test(
    model,
    device,
    tokenizer,
    lines,
    eqn_order,
    num_beam=10,
    num_return_sequences=10,
    test_file=None,
    add_tag=False,
    gentag="generate:",
    ranktag="rank:"
):
    print("gen testing ........")
    model = model.module if hasattr(model, "module") else model
    model.eval()
    acc = acc_all = 0
    total = len(lines)
    rank_samples = []
    rank_samples_list = []
    for i, item in enumerate(tqdm(lines)):
        problem, label, numbers = extract_text_label(item, eqn_order)

        local_samples_list = {}
        local_samples_list["prob"] = problem
        local_samples_list["equations"] = []
        local_samples_list["gt"] = []
        local_samples_list["correct_answer"] = label

        hit_acc = hit_1 = False
        problem = add_tag_tosent(problem=problem, tag=gentag, add_tag=add_tag)
        batch = tokenizer.prepare_seq2seq_batch(
            problem, src_lang=SRC_LANG, return_tensors="pt"
        )
        for k, v in batch.items():
            batch[k] = v.to(device)
        text = model.generate(
            **batch,
            num_beams=num_beam,
            early_stopping=True,
            max_length=100,
            num_return_sequences=num_return_sequences,
        )

        text = tokenizer.batch_decode(text, skip_special_tokens=True)
        text = [clean_text(t) for t in text]

        for idx, candidate in enumerate(text):
            if is_equal(label, candidate, number_filler=True):
                hit_acc = True
                if idx == 0:
                    hit_1 = True
            rank_label = "1" if is_equal(label, candidate, number_filler=True) else "0"
            rank_samples.append([problem, candidate, rank_label, label])

            local_samples_list["equations"].append(candidate)
            local_samples_list["gt"].append(rank_label)
        
        rank_samples_list.append(local_samples_list)

        if hit_acc:
            acc_all += 1
        if hit_1:
            acc += 1

    with open(test_file, "w") as f:
        json.dump(rank_samples_list, f, indent=4)

    return acc, acc_all, total


def main(args):
    global SRC_LANG, TGT_LANG
    SRC_LANG = args.src_lang
    TGT_LANG = args.tgt_lang

    # Manually set the device ids.
    if not args.no_cuda:
        assert (
            torch.cuda.is_available()
        ), "CUDA driver is not available.\
                                            Use --no_cuda to run on CPU."
        device_count = torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

        print("device_id (local_rank): %s" % args.local_rank)
        print(
            "device_count: %s, rank: %s, world_size: %s"
            % (device_count, args.rank, args.world_size)
        )
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    #        args.local_rank = -1

    if args.distributed:
        torch.distributed.init_process_group(
            backend="nccl", world_size=args.world_size, rank=args.rank
        )
    print("init done")
    if args.distributed:
        torch.distributed.barrier()

    if not os.path.exists(args.output_dir) and args.rank == 0:
        os.makedirs(args.output_dir)

    args.n_gpu = torch.cuda.device_count()
    logger.info("args: {}".format(args))

    # Set seed
    set_seed(args)

    # Load pretrained tokenizer
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            args.model_path, do_lower_case=args.do_lower_case
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
        # tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large-ntp-py", do_lower_case=args.do_lower_case)
        print("Tokenizer not found. Please check the model path.")

    new_tokens = [f"#{i}" for i in range(30)]
    tokenizer.add_tokens(new_tokens)

    train(args, tokenizer, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--valid_file",
        default=None,
        type=str,
        required=False,
        help="The input validing data file (a text file).",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        required=True,
        help="The input testing data file (a text file).",
    )
    parser.add_argument("--collect_file", default="collect_file.txt", type=str)

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="The model checkpoint for weights initialization.",
    )

    parser.add_argument("--max_source_length", type=int, default=150)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--regenerate", type=int, default=1)
    parser.add_argument("--rule_negatives", type=int, default=0)
    parser.add_argument("--num_negatives", type=int, default=10)
    parser.add_argument("--use_generate_sample", type=int, default=1)

    ## Other parameters
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )

    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=2.0,
        type=float,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )

    parser.add_argument(
        "--logging_steps", type=int, default=100, help="Log every X updates steps."
    )
    parser.add_argument(
        "--test_per_epoch", type=int, default=5, help="Test every X epochs."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="For distributed training: local_rank"
    )

    parser.add_argument("--data_url", type=str, default="", help="s3 url")
    parser.add_argument("--train_url", type=str, default="", help="s3 url")
    parser.add_argument("--src_lang", default="zh_CN", type=str)
    parser.add_argument("--tgt_lang", default="en_XX", type=str)
    parser.add_argument(
        "--distributed",
        default=False,
        action="store_true",
        help="Use distributed training.",
    )
    parser.add_argument(
        "--dataset_name",
        default="svamp",
        type=str,
        help="Use svamp/math23k dataset.",
        required=True,
    )
    parser.add_argument(
        "--num_seq",
        type=int,
        default=10,
        required=False,
        help="Number of sequences to generate for topk calculation",
    )
    parser.add_argument(
        "--eqn_order",
        default="prefix",
        type=str,
        required=True,
        help="Order of equation to generate",
    )
    parser.add_argument(
        "--data_limit",
        default=-1,
        type=int,
        required=True,
        help="How much data to use for training. -1 for all data.",
    )
    parser.add_argument(
        "--debug_preds",
        default=False,
        action="store_true",
        help="Store predictions in a json file or not",
    )
    parser.add_argument(
        "--fold",
        default=-1,
        type=int,
        help="Which fold to use for training. -1 for all data/SVAMP",
    )
    parser.add_argument(
        "--freeze_seq2seq",
        default=False,
        action="store_true",
        help="Freeze seq2seq model weights and train only ranker",
    )
    parser.add_argument(
        "--remove_invalid_eqns_manually",
        default=False,
        action="store_true",
        help="Remove invalid equations manually from seq2seq generated equations before ranking",
    )
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
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))

    if args.valid_file==None:
        valid_file_given = False
    else:
        valid_file_given = True

    project_name = f"reranker_Tags{args.add_tag}-{Path(args.output_dir).stem}-t5newtrainval_{not valid_file_given}-Freeze_seq2seq{args.freeze_seq2seq}-Manualremove_invalid_eqn{args.remove_invalid_eqns_manually}-{args.dataset_name}-n{args.data_limit}-{args.eqn_order}-src{args.max_source_length}-tgt{args.max_target_length}"

    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%b_%d_%Y")
    args.output_dir = (
        Path(args.output_dir).parent
        / f"{Path(args.output_dir).stem}_{timestamp}_{args.dataset_name}_{args.eqn_order}"
    )
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
    print(args)
    main(args)
