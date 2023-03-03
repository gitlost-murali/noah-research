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

class MyT5ForSequenceClassificationAndGeneration(T5ForConditionalGeneration):
    def __init__(self, modelpath , config, d_model, num_labels, dropout=0.0):
        super().__init__(config)
        self.classification_head = ClassificationHead(
            d_model,
            d_model,
            num_labels,
            dropout,
        )
        self.model = T5ForConditionalGeneration.from_pretrained(modelpath)
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(
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
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert return_dict, "only supports dictionary outputs for this class"
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        if cls_label is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

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

        #mask lm head
        mlm_logits = outputs.logits

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(mlm_logits.view(-1, mlm_logits.shape[-1]), labels.view(-1))

        cls_loss = None
        cls_logits = None
        if cls_label is not None:
            #classification head
            hidden_states = outputs.decoder_hidden_states[-1]  # last hidden state

            eos_mask = decoder_input_ids.eq(self.config.eos_token_id) if decoder_input_ids is not None else input_ids.eq(self.config.eos_token_id)

            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
                :, -1, :
            ]
            cls_logits = self.classification_head(sentence_representation)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            cls_loss = loss_fct(cls_logits.view(-1, self.config.num_labels), cls_label.view(-1))
        return Seq2SeqLMAndClassificationOutput(
            loss=masked_lm_loss,
            logits=mlm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            cls_loss=cls_loss,
            cls_logits=cls_logits
        )


