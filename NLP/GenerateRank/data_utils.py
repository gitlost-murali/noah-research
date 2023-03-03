from utils import infix2postfix, infix2prefix, prefix2infix

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

if __name__ == "__main__":
    test_convert_string()