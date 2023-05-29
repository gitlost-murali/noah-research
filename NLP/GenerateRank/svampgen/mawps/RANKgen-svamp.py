#!/usr/bin/env python
# coding: utf-8

# ## Run the RANKGEN model

# In[1]:


import re
import random
import json
with open("./t5_preds.json", "r") as fh:
    samples_ans = json.load(fh)


# In[2]:


import os
import torch




# In[96]:


from typing import List


# In[4]:


from rankgen import RankGenEncoder

rankgen_encoder = RankGenEncoder("kalpeshk2011/rankgen-t5-xl-all")


# In[131]:


def equation_to_natural_language(equation: str) -> str:
    def tokenize(expression: str) -> List[str]:
        tokens = []
        current_token = ''
        for char in expression:
            if char in ['+', '-', '*', '/', '(', ')']:
                if current_token:
                    tokens.append(current_token.strip())
                    current_token = ''
                tokens.append(char)
            else:
                current_token += char
        if current_token:
            tokens.append(current_token.strip())
        return tokens

    def infix_to_postfix(infix: List[str]) -> List[str]:
        stack = []
        postfix = []
        for i in infix:
            if i == '(':
                stack.append(i)
            elif i == '':
                continue
            elif i == ')':
                while stack[-1] != '(':
                    postfix.append(stack.pop())
                stack.pop()
            elif i in ['+', '-', '*', '/']:
                while stack and stack[-1] != '(' and stack[-1] in ['+', '-', '*', '/']:
                    postfix.append(stack.pop())
                stack.append(i)
            else:
                postfix.append(i)
        while stack:
            postfix.append(stack.pop())
        return postfix

    def postfix_to_natural_language(postfix: List[str]) -> str:
        stack = []
        for token in postfix:
            if token in ['+', '-', '*', '/']:
                right = stack.pop()
                left = stack.pop()
                if token == '+':
                    stack.append(f"({left} plus {right})")
                elif token == '-':
                    stack.append(f"({left} minus {right})")
                elif token == '*':
                    stack.append(f"({left} times {right})")
                elif token == '/':
                    stack.append(f"({left} divided by {right})")
            else:
                stack.append(token)
        return stack.pop()

    infix_tokens = tokenize(equation)
    postfix_tokens = infix_to_postfix(infix_tokens)
    try:
        return postfix_to_natural_language(postfix_tokens)
    except:
        return "Incorrect equation"


# In[132]:


from rich.progress import track


# In[ ]:
def replace_numholders(mwp, equation):
    """
    Replace the number placeholders in the equation with the actual numbers
    Example: number0 * (number1 + number0) -> 2 * (3 + 2)
    """
    # count the number of placeholders using regex
    num_placeholders = (re.findall(r"number\d+", equation))
    # only consider unique numbers
    num_placeholders = list(set(num_placeholders))

    number_mapper = dict()
    for i in range(len(num_placeholders)):
        num = random.randint(1, 50)
        equation = equation.replace(f"number{i}", str(num))
        mwp = mwp.replace(f"number{i}", str(num))
        number_mapper[f"number{i}"] = str(num)
    
    return mwp, equation, number_mapper

def replace_numholds_eqns(equations, number_mapper):
    result_equations = []
    for eq in equations:
        for k,v in number_mapper.items():
            eq = eq.replace(k,v)
        
        result_equations.append(eq)
    
    return result_equations


math_eqn_acc = 0
natural_eqn_acc = 0
number_eqn_acc = 0
naturalnumber_eqn_acc = 0
for item in track(samples_ans, total = len(samples_ans)):
    numberitem = dict()
    numberitem['prob'], numberitem['correct_answer'], numbermapper = replace_numholders(item['prob'], item['correct_answer'])
    numberitem['equations'] = replace_numholds_eqns(item['equations'], numbermapper)

    prefix_vectors = rankgen_encoder.encode([item["prob"]+" Give the answer in equation form"], vectors_type="prefix")
    suffix_vectors = rankgen_encoder.encode(item["equations"], vectors_type="suffix")
    natural_suffix_vectors = rankgen_encoder.encode([equation_to_natural_language(it) for it in item["equations"]], vectors_type="suffix")

    num_prefix_vectors = rankgen_encoder.encode([numberitem["prob"]+" Give the answer in equation form"], vectors_type="prefix")
    num_suffix_vectors = rankgen_encoder.encode(numberitem["equations"], vectors_type="suffix")
    num_natural_suffix_vectors = rankgen_encoder.encode([equation_to_natural_language(it) for it in numberitem["equations"]], vectors_type="suffix")


    scores = torch.matmul(prefix_vectors["embeddings"], suffix_vectors["embeddings"].transpose(1,0))
    best_eqnidx_rankgen = torch.argmax(scores)
    if item["gt"][best_eqnidx_rankgen] == 1:  math_eqn_acc+=1

    natural_scores = torch.matmul(prefix_vectors["embeddings"], natural_suffix_vectors["embeddings"].transpose(1,0))
    best_eqnidx_rankgen_nat = torch.argmax(natural_scores)
    if item["gt"][best_eqnidx_rankgen_nat] == 1: natural_eqn_acc+=1    

    number_scores = torch.matmul(num_prefix_vectors["embeddings"], num_suffix_vectors["embeddings"].transpose(1,0))
    num_best_eqnidx_rankgen = torch.argmax(number_scores)
    if item["gt"][num_best_eqnidx_rankgen] == 1:  number_eqn_acc+=1

    number_natural_scores = torch.matmul(num_prefix_vectors["embeddings"], num_natural_suffix_vectors["embeddings"].transpose(1,0))
    numnat_best_eqnidx_rankgen_nat = torch.argmax(number_natural_scores)
    if item["gt"][numnat_best_eqnidx_rankgen_nat] == 1: naturalnumber_eqn_acc+=1    


# In[ ]:


print(f"Math eqn accuracy is {math_eqn_acc}")

print(f"Natural eqn accuracy is {natural_eqn_acc}")

print(f"Numbers in eqn accuracy is {number_eqn_acc}")

print(f"Natural Numbers in eqn accuracy is {naturalnumber_eqn_acc}")
# In[ ]:




