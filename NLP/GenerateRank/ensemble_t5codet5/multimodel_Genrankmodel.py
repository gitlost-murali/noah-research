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

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration
from transformers.models.mbart.modeling_mbart import (
    shift_tokens_right
)
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple


class ClassificationHead(nn.Module):
    def __init__(self, input_size=1024, hidden_size=1024, output_size=2, dropout=0.0):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.dense(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@dataclass
class Seq2SeqLMAndClassificationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cls_loss: Optional[torch.FloatTensor] = None
    cls_logits: torch.FloatTensor = None

class ABC():
    pass

class T5_codeT5_ForSequenceClassificationAndGeneration(T5ForConditionalGeneration):
    def __init__(self, modelpath, modelcode_path, config, t5model_dim,
                 codet5model_dim, num_labels, dropout=0.0):
        super().__init__(config)
        self.classification_head = ClassificationHead(
            t5model_dim + codet5model_dim, # input dim
            t5model_dim + codet5model_dim, # hidden dim
            num_labels,
            dropout,
        )
        self.model = T5ForConditionalGeneration.from_pretrained(modelpath)
        self.codemodel = T5ForConditionalGeneration.from_pretrained(modelcode_path)
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def generate_t5(self, **kwargs):
        # call generate function of the parent class T5ForConditionalGeneration
        return self.model.generate(**kwargs)

    def generate_codet5(self, **kwargs):
        # call generate function of the parent class T5ForConditionalGeneration
        return self.codemodel.generate(**kwargs)

    def forward_pass(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        cls_label=None, #this is new
        use_cache=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
        freeze_seq2seq=False,
        **kwargs,):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert return_dict, "only supports dictionary outputs for this class"

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        return outputs

    def forward(
        self,
        t5_batch = None,
        codet5_batch = None,
        freeze_seq2seq=False,
        return_dict=None,
        **kwargs,
    ):

        labels = t5_batch.get("labels", None)
        cls_label = codet5_batch["cls_label"]

        if cls_label is not None:
            use_cache = False

        outputs_t5 = self.forward_pass(**t5_batch)
        outputs_codet5 = self.forward_pass(**codet5_batch)

        #mask lm head
        mlm_logits = outputs_t5.logits
        mlm_logits_codet5 = outputs_codet5.logits

        masked_lm_loss = None
        if labels is not None and freeze_seq2seq is False:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss_t5 = loss_fct(mlm_logits.view(-1, mlm_logits.shape[-1]), labels.view(-1))

            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss_codet5 = loss_fct(mlm_logits_codet5.view(-1, mlm_logits_codet5.shape[-1]), labels.view(-1))

            masked_lm_loss = masked_lm_loss_t5 + masked_lm_loss_codet5

        cls_loss = None
        cls_logits = None
        if cls_label is not None:
            #classification head
            hidden_states_t5 = outputs_t5.decoder_hidden_states[-1]  # last hidden state
            hidden_states_codet5 = outputs_codet5.decoder_hidden_states[-1]  # last hidden state



            eos_mask_t5 = t5_batch["decoder_input_ids"].eq(self.config.eos_token_id) if t5_batch["decoder_input_ids"] is not None else t5_batch["input_ids"].eq(self.config.eos_token_id)
            if len(torch.unique(eos_mask_t5.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            sentence_representation_t5 = hidden_states_t5[eos_mask_t5, :].view(hidden_states_t5.size(0), -1, hidden_states_t5.size(-1))[
                :, -1, :
            ]

            eos_mask_codet5 = codet5_batch["decoder_input_ids"].eq(self.config.eos_token_id) if codet5_batch["decoder_input_ids"] is not None else codet5_batch["input_ids"].eq(self.config.eos_token_id)
            if len(torch.unique(eos_mask_codet5.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            sentence_representation_codet5 = hidden_states_codet5[eos_mask_codet5, :].view(hidden_states_codet5.size(0), -1, hidden_states_codet5.size(-1))[
                :, -1, :
            ]
            sentence_representation = torch.cat((sentence_representation_t5,
                                                sentence_representation_codet5), dim=1)

            if freeze_seq2seq:
                sentence_representation = sentence_representation.detach()
            cls_logits = self.classification_head(sentence_representation)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            cls_loss = loss_fct(cls_logits.view(-1, self.config.num_labels), cls_label.view(-1))
        
        return Seq2SeqLMAndClassificationOutput(
            loss=masked_lm_loss,
            logits=mlm_logits,
            past_key_values=outputs_t5.past_key_values,
            decoder_hidden_states=outputs_t5.decoder_hidden_states,
            decoder_attentions=outputs_t5.decoder_attentions,
            cross_attentions=outputs_t5.cross_attentions,
            encoder_last_hidden_state=outputs_t5.encoder_last_hidden_state,
            encoder_hidden_states=outputs_t5.encoder_hidden_states,
            encoder_attentions=outputs_t5.encoder_attentions,
            cls_loss=cls_loss,
            cls_logits=cls_logits
        )


