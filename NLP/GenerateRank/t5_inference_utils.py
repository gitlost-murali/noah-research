
from utils import clean_text, prefix2infix, add_to_topk_accuracylist, is_equal, is_equal_svamp
from torch.utils.data import Dataset
from tqdm import tqdm
import time
import torch

class InferDataset(Dataset):
    """
    Ideally, should work with any dataset.
    Just pass the inference lines to the constructor.
    """
    def __init__(self, lines, labels, **kwargs):
        self.lines = lines
        self.labels = labels
        # create attribute for all kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        # print all attributes
        return dict((k, self.__dict__[k][item]) for k in self.__dict__)

def batch_test(model, tokenizer,  device, lines, dataset_name,
        num_beam=10, num_return_sequences=1, eqn_order="prefix",
        max_target_length=100, batch_size=8):
    model = model.module if hasattr(model, "module") else model
    model.eval()
    acc = 0
    total = len(lines)
    topk_acc_list = {(k+1): 0 for k in range(num_return_sequences)}

    if dataset_name == "svamp":
        # apply prefix2infix function to all labels
        if eqn_order == "infix":
            equations = lines.Equation.apply(lambda x: prefix2infix(x.split(" ")))
        else:
            equations = lines.Equation
        problems, labels, numbers_list = lines.Question, equations, lines.Numbers
        test_dataset = InferDataset(problems, labels, numbers=numbers_list)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=batch_size,
                                                        drop_last=False)
        epoch_iterator = tqdm(test_dataloader, desc="Inferring...")
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # batch inference
            texts = batch_inference(model, tokenizer, device, inputs["lines"],
                                   num_beam, num_return_sequences, max_target_length)

            labels, numbers_list = inputs["labels"], inputs["numbers"]
            for candidatenum, (label, numbers, candidates_list) in enumerate(zip(labels, numbers_list, texts)):
                for candidate in candidates_list:
                    if is_equal_svamp(label, candidate, numbers.split(), order=eqn_order):
                        acc += 1
                        topk_acc_list = add_to_topk_accuracylist(candidatenum, topk_acc_list, num_return_sequences)
                        break
    else:
        problems = labels = []
        for i, line in enumerate(tqdm(lines)):
            problem, label = line.strip().split("\t")
            problems.append(problem)
            labels.append(label)
        test_dataset = InferDataset(problems, labels)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=batch_size,
                                                        drop_last=False)
        epoch_iterator = tqdm(test_dataloader, desc="Inferring...")
        train_start_time = time.time()
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # batch inference
            texts = batch_inference(model, tokenizer, device, inputs["lines"],
                                   num_beam, num_return_sequences, max_target_length)

            labels = inputs["labels"]
            for candidatenum, label, candidates_list in enumerate(zip(labels, texts)):
                for candidate in candidates_list:
                    if is_equal(label, candidate):
                        acc += 1
                        topk_acc_list = add_to_topk_accuracylist(candidatenum, topk_acc_list, num_return_sequences)
                        break
    return topk_acc_list, total

def batch_inference(model, tokenizer, device, problem, num_beam=10, num_return_sequences=1, max_target_length=100, SRC_LANG="en_XX"):
    batch = tokenizer.prepare_seq2seq_batch(problem, src_lang=SRC_LANG, return_tensors="pt")
    for k,v in batch.items():
        batch[k] = v.to(device)

    # extra_details = {"decoder_start_token_id":tokenizer.lang_code_to_id["en_XX"]}
    text = model.generate(**batch,
                            num_beams=num_beam, early_stopping=True, max_length=max_target_length,
                            num_return_sequences=num_return_sequences)

    text = tokenizer.batch_decode(text, skip_special_tokens=True)
    text = [clean_text(t) for t in text]

    texts = []
    idx = 0
    for idx in range(batch["input_ids"].shape[0]):
        beam = text[idx*num_return_sequences: (idx+1)*num_return_sequences]
        texts.append(beam)
    return texts

