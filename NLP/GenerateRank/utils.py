# coding=utf-8
# Copyright 2022 Huawei Technologies Co., Ltd.
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

import random
import math
import re

def remove_space_and_bracket(text):
    text = text.strip().replace(" ", "")
    text = re.sub("[\[\{]", "(", text)
    text = re.sub("[\]\}]", ")", text)
    return text

def split_eq(text):
    text = re.split(r"([=\+\-\*\/\{\}\(\)\[\]\^])", text)
    return [x for x in text if x]

def get_test_nums():
    #mapping from symbol numbers to real numbers
    nums = {"#_pi": 3.14, "PI": 3.14}
    for i in range(10):
        nums[str(i)] = float(i)
    for i in range(50):
        nums[f"#{i}"] = random.random()
    return nums

def calculate_eval(equation, nums):
    op_list = ["+", "-", "*", "/", "(", ")", "^"]
    equation = clean_text(equation).split(" ")
    try:
        for i, e in enumerate(equation):
            if e not in op_list:
                if e in nums:
                    equation[i] = str(nums[e])
                else:
                    equation[i] = e
            if equation[i][-1] == "%":
                equation[i] = f"( {equation[i][:-1]} / 100 )"

        after_number_exp = " ".join(equation)
        assert not '#' in after_number_exp
        after_number_exp = after_number_exp.replace("^", "**")
        ans = eval(after_number_exp)
    except:
        return None
    return ans

from typing import List

# Convert prefix expression to infix expression
def prefix2infix(prefix: List[str]) -> List[str]:
    """Example: 
    prefix = ['+', '1', '2']
    infix = ['(', '1', '+', '2', ')']"""
    stack = []
    for i in range(len(prefix)-1, -1, -1):
        if prefix[i] in ['+', '-', '*', '/']:
            op1 = stack.pop()
            op2 = stack.pop()
            stack.append('(' +" "+ op1 +" "+ prefix[i] +" "+ op2 +" "+ ')')
        else:
            stack.append(prefix[i])
    return stack.pop()

def calculate_eval_svamp(equation, nums, order = "prefix"):
    op_list = ["+", "-", "*", "/", "(", ")", "^"]
    try:
        if order=="prefix": equation = prefix2infix(equation.split(" ")).split(" ")
        else: equation = clean_text(equation).split(" ")
        for i, e in enumerate(equation):
            if e not in op_list:
                if e in nums:
                    equation[i] = str(nums[e])
                else:
                    equation[i] = e
            if equation[i][-1] == "%":
                equation[i] = f"( {equation[i][:-1]} / 100 )"

        after_number_exp = " ".join(equation)
        assert not '#' in after_number_exp
        after_number_exp = after_number_exp.replace("^", "**")
        ans = eval(after_number_exp)
    except:
        return None
    return ans

def is_equal(label, text):
    for test_times in range(3):
        failed = 0
        label_ans = None
        while label_ans is None:
            failed += 1
            if failed == 5:
                return False
            nums = get_test_nums()
            label_ans = calculate_eval(label, nums)
        text_ans = calculate_eval(text, nums)
        try:
            if text_ans is None or abs(text_ans - label_ans) > 1e-5:
                return False
        except:
            return False
    return True

def is_equal_svamp(label, text, numbers, order = "prefix"):
    nums = {"#_pi": 3.14, "PI": 3.14}
    for ix, num in enumerate(numbers):
        nums[f"number{ix}"] = num
    for test_times in range(3):
        failed = 0
        label_ans = None
        while label_ans is None:
            failed += 1
            if failed == 5:
                return False
            label_ans = calculate_eval_svamp(label, nums, order=order)
        text_ans = calculate_eval_svamp(text, nums, order=order)
        try:
            if text_ans is None or abs(text_ans - label_ans) > 1e-5:
                return False
        except:
            return False
    return True

def clean_text(text):
    splited_text = split_eq(remove_space_and_bracket(text))
    bracket = 0
    for i, s in enumerate(splited_text):
        if s=="(":
            bracket += 1
        elif s == ")":
            bracket -= 1
            if bracket < 0:
                return " ".join(splited_text[:i])
    return " ".join(splited_text)

# function to calculate top-k accuracy given the index at which answer is correct
def add_to_topk_accuracylist(candidate_number, topk_acc_list, topk):
    for k in range(candidate_number, topk, 1):
        topk_acc_list[k+1] += 1
    return topk_acc_list
