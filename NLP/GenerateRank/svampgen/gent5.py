import torch
import time
from tqdm import tqdm


train_file = "../data/mawps_asdiv-a_svamp/trainset_nodup.json"
model_path = "/scratch/s5397294/svamp_t5_batch_output/generator_Mar_31_2023_svamp_infix/saved_model/"
outfile = "traint5_preds.json"

from transformers import T5Config, T5Tokenizer

import sys
sys.path.append("../t5_codet5_based/")
from t5_GenerateRankModel import MyT5ForSequenceClassificationAndGeneration

sys.path.append("../")
from utils import read_json, clean_text, is_equal
from data_utils import extract_text_label

device = torch.device("cuda")

from torch.utils.data import Dataset

class GeneralDataset(Dataset):
    """
    Ideally, should work with any dataset.
    Just pass the inference lines to the constructor.
    """
    def __init__(self, **kwargs):
        # create attribute for all kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        for k in self.__dict__:
            return len(self.__dict__[k])
        return -1

    def __getitem__(self, item):
        # print all attributes
        return dict((k, self.__dict__[k][item]) for k in self.__dict__)

def generate_sample(model, tokenizer, dataloader, device):
    """
    generate negative samples using the model for revise training
    """
    samples = []
    model = model.module if hasattr(model, "module") else model
    model.eval()
    beam_size = 10
    for data in tqdm(dataloader):
        prob, label = data["prob"], data["label"]
        gen_prob = prob
        batch = tokenizer.prepare_seq2seq_batch(gen_prob, return_tensors="pt")
        for k, v in batch.items():
            batch[k] = v.to(device)

        text = model.generate(
            **batch,
            num_beams=beam_size,
            early_stopping=True,
            max_length=64,
            num_return_sequences=beam_size,
        )  # batch * 10, len
        text = tokenizer.batch_decode(text, skip_special_tokens=True)
        text = [clean_text(t) for t in text]

        label = [clean_text(t) for t in label]

        idx = 0
        
        samples_list = []

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
            
            samples_list.append(local_samples_list)

            idx += 1

    return samples_list


# open a json file and read it where text is in "text" key and infix equation is in "template_equ" key
data_limit = -1
batch_size = 16
eqn_order = "infix"
# "generator_Mar_03_2023_svamp_infix/saved_model/"

data = read_json(train_file)
lines = []
labels = []
numbers_list = []
for i, item in enumerate(tqdm(data, desc="Prepare train data")):
    goal, proof, numbers = extract_text_label(item, eqn_order)
    lines.append(goal)
    labels.append(proof)
    numbers_list.append(numbers)
    if data_limit > 0 and i > data_limit:
        break
raw_train_dataset = GeneralDataset(
    prob=lines, label=labels, numbers=numbers_list
)

extra_args = {}
raw_train_dataloader = torch.utils.data.DataLoader(
    raw_train_dataset,
    batch_size=batch_size,
    drop_last=False,
    **extra_args,
    )

tokenizer = T5Tokenizer.from_pretrained(
    model_path, do_lower_case=False
)

print(f"load model from {model_path}")
config = T5Config.from_pretrained(model_path)
config.num_labels = 2
config.id2label = {"0": "LABEL_0", "1": "LABEL_1"}
config.label2id = {"LABEL_0": 0, "LABEL_1": 1}

model = MyT5ForSequenceClassificationAndGeneration(
    modelpath= model_path, config=config, d_model=config.d_model, num_labels=2
)
model.resize_token_embeddings(len(tokenizer))
print("model load done")

config = model.config
model.to(device)

# model size
size = 0
for n, p in model.named_parameters():
    size += p.nelement()
print("Total parameters: {}".format(size))

# Training
# generate samples for revise

print("start generate samples")
gen_start_time = time.time()
samples_ans = generate_sample(
    model, tokenizer, raw_train_dataloader, device
)
print(f"finish generate samples in {time.time() - gen_start_time}")

import json
with open(outfile, "w") as fh:
    json.dump(samples_ans, fh, indent=4)