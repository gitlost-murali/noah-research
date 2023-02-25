from utils import infix2postfix, infix2prefix

def extract_equation(item, eqn_order):
    goal= item["text"]
    numbers = " ".join([str(st) for st in item["num_list"]])
    if eqn_order == "infix":
        proof = item["template_equ"]
    elif eqn_order == "postfix":
        proof = item["post_equ"]
    elif eqn_order == "prefix":
        proof = infix2postfix(item["template_equ"])
        proof = " ".join(proof)
    else:
        raise ValueError("Unknown equation order: {}".format(eqn_order))
    
    # convert temp variables to unique numbers
    proof = convert_temp_vars(proof)
    return goal, proof, numbers

def extract_equation_svamp(item, eqn_order):
    goal = item["Question"]
    infix_eqn = item["Equation"]
    numbers = item["Numbers"]
    if eqn_order == "infix":
        proof = infix_eqn
    elif eqn_order == "postfix":
        proof = infix2postfix(infix_eqn)
        proof = " ".join(proof)
    elif eqn_order == "prefix":
        proof = infix2prefix(infix_eqn)
        proof = " ".join(proof)
    else:
        raise ValueError("Unknown equation order: {}".format(eqn_order))
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
    # find all temp variables in the string
    temp_vars = re.findall(r'temp_[a-z]', string)

    # create a dictionary mapping each temp variable to a unique number
    var_map = {var: f'number_{i}' for i, var in enumerate(temp_vars)}

    # replace each temp variable with its corresponding number
    for var, num in var_map.items():
        string = string.replace(var, num)

    return string