import tqdm
import torch
from transformers import T5Tokenizer, T5EncoderModel, AutoModel

class RankGenEncoder():
    def __init__(self, model_path, max_batch_size=32, model_size=None, cache_dir=None):
        assert model_path in ["kalpeshk2011/rankgen-t5-xl-all", "kalpeshk2011/rankgen-t5-xl-pg19", "kalpeshk2011/rankgen-t5-base-all", "kalpeshk2011/rankgen-t5-large-all"]
        self.max_batch_size = max_batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_size is None:
            if "t5-large" in model_path or "t5_large" in model_path:
                self.model_size = "large"
            elif "t5-xl" in model_path or "t5_xl" in model_path:
                self.model_size = "xl"
            else:
                self.model_size = "base"
        else:
            self.model_size = model_size

        self.tokenizer = T5Tokenizer.from_pretrained(f"google/t5-v1_1-{self.model_size}", cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, inputs, vectors_type="prefix", verbose=False, return_input_ids=False):
        tokenizer = self.tokenizer
        max_batch_size = self.max_batch_size
        if isinstance(inputs, str):
            inputs = [inputs]
        if vectors_type == 'prefix':
            inputs = ['pre ' + input for input in inputs]
            max_length = 512
        else:
            inputs = ['suffi ' + input for input in inputs]
            max_length = 128

        all_embeddings = []
        all_input_ids = []
        for i in tqdm.tqdm(range(0, len(inputs), max_batch_size), total=(len(inputs) // max_batch_size) + 1, disable=not verbose, desc=f"Encoding {vectors_type} inputs:"):
            tokenized_inputs = tokenizer(inputs[i:i + max_batch_size], return_tensors="pt", padding=True)
            for k, v in tokenized_inputs.items():
                tokenized_inputs[k] = v[:, :max_length]
            tokenized_inputs = tokenized_inputs.to(self.device)
            with torch.inference_mode():
                batch_embeddings = self.model(**tokenized_inputs)
            all_embeddings.append(batch_embeddings)
            if return_input_ids:
                all_input_ids.extend(tokenized_inputs.input_ids.cpu().tolist())
        return {
            "embeddings": torch.cat(all_embeddings, dim=0),
            "input_ids": all_input_ids
        }

