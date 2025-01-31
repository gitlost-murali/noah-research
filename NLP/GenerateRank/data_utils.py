from utils import infix2postfix, infix2prefix, prefix2infix, read_json, write_2_json
import random
from random import shuffle
import os

def extract_equation(item, eqn_order):
    goal= item["text"]
    numbers = " ".join([str(st) for st in item["num_list"]])
    if eqn_order == "infix":
        proof = item["template_equ"]
    elif eqn_order == "postfix":
        proof = item["post_equ"]
    elif eqn_order == "prefix":
        proof = infix2prefix(item["template_equ"].lower().replace("x = ","").split(" "))
        proof = " ".join(proof)
    else:
        raise ValueError("Unknown equation order: {}".format(eqn_order))
    
    proof = proof.lower().replace("x = ","")
    # convert temp variables to unique numbers
    goal = convert_temp_vars(goal)
    proof = convert_temp_vars(proof)
    return goal, proof, numbers

def extract_equation_svamp(item, eqn_order):
    goal = item["Question"]
    prefix_eqn = item["Equation"].lower().replace("x = ","")
    numbers = item["Numbers"]
    if eqn_order == "prefix":
        proof = prefix_eqn
    elif eqn_order == "postfix":
        infix_eqn = prefix2infix(prefix_eqn.split(" "))
        proof = infix2postfix(infix_eqn.split(" "))
        proof = " ".join(proof)
    elif eqn_order == "infix":
        proof = prefix2infix(prefix_eqn.split(" "))
    else:
        raise ValueError("Unknown equation order: {}".format(eqn_order))
    proof = proof.lower().replace("x = ","")
    # convert temp variables to unique numbers
    goal = convert_temp_vars(goal)
    proof = convert_temp_vars(proof)
    return goal, proof, numbers

def extract_text_label(item, eqn_order):
    if "new_text" in item:
        goal, proof, numbers = extract_equation(item, eqn_order)
    else:
        goal, proof, numbers = extract_equation_svamp(item, eqn_order)
    return goal, proof, numbers

import re
def convert_temp_vars(string):
    """
    Convert all temp variables in a string to unique numbers.
    ex: temp_a + temp_b + temp_c -> number_0 + number_1 + number_2
    """
    # Split the input string by spaces to create a list of words
    words = string.split()
    # Use a list comprehension to map each temp_x word in the input string to its corresponding number
    numbers = [f"number{ord(word[-1])-97}" if word.startswith("temp_") else word for word in words]
    # Join the list of numbers into a single string, separated by spaces
    output_string = ' '.join(numbers)
    return output_string

def test_convert_string():
    assert convert_temp_vars("temp_a") == "number0"
    assert convert_temp_vars("temp_b temp_a temp_a temp_b") == "number1 number0 number0 number1"
    assert convert_temp_vars("this is a test") == "this is a test"
    assert convert_temp_vars("temp_z temp_y temp_z") == "number25 number24 number25"
    assert convert_temp_vars("") == ""

def remove_invalid_equations(collect_lines):
    """
    Remove equations that are invalid for calculations (e.g. 1/0 or incorrect number of parantheses)
    """
    only_valid = []
    for line in collect_lines:
        # regex to get number0, number1, number2 from "line"
        numbers_in_prob = re.findall(r"number\d+", line)
        numbers_to_fill = [str(random.randint(0, 10)) for _ in range(len(numbers_in_prob))]
        numbers2fill = dict(zip(numbers_in_prob, numbers_to_fill)) # map number0 -> 5, number1 -> 3, etc.
        gen_eq = line.split("\t")[1]
        for key, value in numbers2fill.items(): gen_eq = gen_eq.replace(key, value)
        try:
            _ = float(eval(gen_eq))
            only_valid.append(line)
        except:
            pass
    return only_valid

import re

# Function to check if the infix equation is valid
def check_valid_equation(eqn):
    """
    (2/1) -> True
    (2/0) -> True
    (2/1) + (2/0) -> True
    (2/1) ++ (2/0) -> False
    2 -* 3 -> False
    2 + 3 -> True
    2 % 2 -> True
    2 + 3 + 4 -> True
    4 % 1 -> True
    """
    eqn = eqn.replace("%", "/")
    # Check for invalid characters
    if re.search(r"[^0-9\+\-\*\/\(\)\s]", eqn): return False
    # Check for invalid number of parantheses
    if eqn.count("(") != eqn.count(")"): return False
    # Check for invalid number of operators
    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", eqn): return True
    return False

def check_invalid_numberinst(eqn, numbers_in_prob):
    """
    Check for cases like number01 or number012343
    Problem is, if you do replace "number0" in "number01" with 8, 
    you get 81 which is not a valid answer
    """
    numbers_in_eqn = re.findall(r"\S*number\S+", eqn)
    for num in numbers_in_eqn:
        if num not in numbers_in_prob: return True
    return False

def remove_invalid_equations_genranktags(collect_lines):
    """
    Remove equations that are invalid for calculations (e.g. 1/0 or incorrect number of parantheses)
    """
    only_valid = []
    for item in collect_lines:
        # regex to get number0, number1, number2 from "line"
        line = item["prob"]
        numbers_in_prob = re.findall(r"number\d+", line)
        numbers_to_fill = [str(random.randint(0, 10)) for _ in range(len(numbers_in_prob))]
        numbers2fill = dict(zip(numbers_in_prob, numbers_to_fill)) # map number0 -> 5, number1 -> 3, etc.
        gen_eqns = item["equations"]
        gts = item["gt"]
        filtered_eqns = []
        filtered_eqns_gts = []
        for gen_eq, gt in zip(gen_eqns, gts):
            gen_eq_eval = gen_eq
            try:
                assert not check_invalid_numberinst(gen_eq_eval, numbers_in_prob)
                for key, value in numbers2fill.items(): gen_eq_eval = gen_eq_eval.replace(key, value)
                assert "#" not in gen_eq_eval
                assert check_valid_equation(gen_eq_eval)
                _ = float(eval(gen_eq_eval))
                filtered_eqns.append(gen_eq)
                filtered_eqns_gts.append(gt)
            except:
                pass
        item["old_equations"] = gen_eqns
        item["equations"] = filtered_eqns
        item["gt"] = filtered_eqns_gts
        only_valid.append(item)

    return only_valid

def create_newtrainval_splits(args, split_ratio=0.85):
    """
    Create new train and val splits from the original train file.
    """
    print("Creating new train and val splits from the original train file...")
    data = read_json(args.train_file) # read original train file
    shuffle(data) # shuffle the data
    train_data = data[:int(len(data)*split_ratio)] # split into train and val
    val_data = data[int(len(data)*split_ratio):] # split into train and val
    train_filepath = os.path.join(args.output_dir, "new_train.json") # write to new files
    val_filepath = os.path.join(args.output_dir, "new_val.json") # write to new files
    write_2_json(train_data, train_filepath) # write to new files
    write_2_json(val_data, val_filepath) # write to new files
    print("New train and val splits created and saved to {} and {}.".format(train_filepath, val_filepath))
    return train_filepath, val_filepath

def add_tag_tosent(problem, tag, add_tag=False):
    """
    Add a tag to a problem.
    """
    if add_tag:
        problem = tag + " " + problem
    return problem

def add_rank_eqns(problem, list_of_eqns, add_tag=False):
    """
    Add a tag to a problem.
    """
    if add_tag:
        problem = problem + ". Re-rank the following equations to select the best equation: " + " ; ".join(list_of_eqns)
    return problem



if __name__ == "__main__":
    test_convert_string()