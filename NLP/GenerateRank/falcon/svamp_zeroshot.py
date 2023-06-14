import re
import sys
import torch
import random
import argparse
from tqdm import tqdm
import transformers

sys.path.append("../")
from utils import read_json, is_equal_svamp
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
    
def sample_data(data, k):
    return random.sample(data, k)

def generate_single_prefix_prompt(instance):
    local_goal, local_proof, local_pred_eqns = instance["prob"], instance["correct_answer"], instance["equations"]
    prompt_text = f'Task: Solve the following math word problem: {local_goal}. Give the final answer as an equation in infix order. Be aware of the paranthesis when BODMAS issues will arrive. Answer: {local_proof} <eos>\n'
    return prompt_text

def generate_prefix_prompt(data, k):
    instances = sample_data(data, k)
    prefix_prompt = ''
    for instance in instances:
        prefix_prompt += generate_single_prefix_prompt(instance)
    return prefix_prompt

def main(
    incontext_samples: int = 5,
    num_samples: int = 1,
    max_new_tokens: int = 64,
    top_k: int = 200,
    temperature: float = 0.1,
    model_path = "/scratch/s5397294/falcon-7b/",
    trainfile = "../svampgen/svamp/traint5_preds.json",
    testfile = "../svampgen/svamp/t5_preds.json",
    quantize = False,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        model path: The checkpoint path to load.
    """
    data = read_json(testfile)
    traindata = read_json(trainfile)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    prefix_prompts = [generate_prefix_prompt(traindata, incontext_samples) for _ in range(len(data))]
    acc = 0
    for i, item in enumerate(tqdm(data, desc="Prepare train data")):
        goal, proof, pred_eqns = item["prob"], item["correct_answer"], item["equations"]
        # regex to get number0, number1, etc from goal
        number_instances = re.findall(r"number\d+", goal)
        number_instances = list(set(number_instances))
        # generate random numbers for each number instance
        numbers = [str(random.randint(0, 100)) for _ in range(len(number_instances))]

        input_text = f'Task: Solve the following math word problem: {goal}. Give the final answer as an equation in infix order. Be aware of the paranthesis when BODMAS issues will arrive. Answer: '
        prefix_prompt = prefix_prompts[i]
        input_text += prefix_prompt

        sequences = pipeline(
            input_text,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_eqn = sequences[0]["generated_text"].strip()
        res = is_equal_svamp(proof, generated_eqn, numbers, "infix")
        if res:
            acc += 1
        print(f"Matched ones are {acc}  out of {i+1}, {generated_eqn} \n")
        print("*"*50)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--incontext_samples", type=int, default=5)
    argparser.add_argument("--num_samples", type=int, default=1)
    argparser.add_argument("--max_new_tokens", type=int, default=64)
    argparser.add_argument("--top_k", type=int, default=200)
    argparser.add_argument("--temperature", type=float, default=0.1)
    argparser.add_argument("--model_path", type=str, default="/scratch/s5397294/falcon-7b/")
    argparser.add_argument("--trainfile", type=str, default="../svampgen/svamp/traint5_preds.json")
    argparser.add_argument("--testfile", type=str, default="../svampgen/svamp/t5_preds.json")
    argparser.add_argument("--quantize", type=bool, default=False)
    args = argparser.parse_args()
    main(**vars(args))
