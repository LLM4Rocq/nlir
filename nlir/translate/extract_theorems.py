import re
import os
import json
import sys

def extract_lean(file_path: str):
    """
    Extracts all theorems and their proofs from a Lean file.

    Args:
        file_path (str): Path to the Lean file.

    Returns:
        list[(str, str)]: List of pairs of strings, a name and the corresponding theorem.
    """
    # Regular expression to match theorems and their proofs
    theorem_pattern = re.compile(
        r"theorem\s(?P<name>\S+)(?:(?!theorem)[\S\s])*end",
        re.DOTALL
    )

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Find all matches of theorems
    theorems = theorem_pattern.finditer(content)

    # Save theorems with their names
    theorems = [(match.group('name'), match.group(0)) for match in theorems]
    theorems.sort()

    return theorems

def extract_isabelle(directory: str):
    """
    Extracts all theorems and their proofs from a directory containing Isabelle files for each theorem.

    Args:
        directory (str): Path to the directory.

    Returns:
        list[(str, str)]: List of pairs of strings, a name and the corresponding theorem.
    """

    theorems = []

    # Regular expression to match theorems and their proofs
    theorem_pattern = re.compile(
        r"theory(?:[\S\s]*)end",
        re.DOTALL
    )

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Ensure it's a file (skip directories)
        if os.path.isfile(file_path):
            # Extract theorem name (filename without extension)
            name = os.path.splitext(filename)[0]

            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                content = theorem_pattern.findall(content)[0]

            # Add the theorem to the list
            theorems.append((name, content))

    theorems.sort()

    return theorems

def extract_informal(directory: str):
    """
    Extracts all theorems informal description from a directory containing JSON files for each theorem.

    Args:
        directory (str): Path to the directory.

    Returns:
        list[(str, str)]: List of pairs of strings, a name and the corresponding theorem.
    """

    theorems = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Ensure it's a file (skip directories)
        if os.path.isfile(file_path):

            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = json.load(file)
                name = content["problem_name"]
                theorem = content["informal_statement"]
                # proof = content["informal_proof"]

            # Add the theorem to the list
            theorems.append((name, theorem))

    theorems.sort()

    return theorems

def extract_all(directory: str) :
    """
    Extracts the Lean, Isabelle and informal description version of all theorems and their proofs from a directory matching the structure of the Facebook's miniF2F git repository : https://github.com/facebookresearch/miniF2F/tree/main.

    Args:
        directory (str): Path to the directory.

    Returns:
        dict[str, dict[str, dict[str, str]]]: A dictionnary containing the valid and test sets of theorems.
        The sets of theorems are themselves dictionnaries containing the Lean, Isabelle and informal version for each theorem.
    """
    # Extract all directories
    lean_directory = os.path.join(directory, "lean/src")
    if not os.path.isdir(lean_directory):
        print("Error: the parent directory has not the right structure, you should have a './lean/src' directory inside of it.")
        sys.exit(1)
    isab_directory = os.path.join(directory, "isabelle")
    if not os.path.isdir(isab_directory):
        print("Error: the parent directory has not the right structure, you should have a './isabelle' directory inside of it.")
        sys.exit(1)
    info_directory = os.path.join(directory, "informal")
    if not os.path.isdir(info_directory):
        print("Error: the parent directory has not the right structure, you should have a './informal' directory inside of it.")
        sys.exit(1)

    # Extract all valid lean and isabelle theorems
    valid_lean_th = extract_lean(os.path.join(lean_directory, "valid.lean"))
    valid_isab_th = extract_isabelle(os.path.join(isab_directory, "valid"))
    valid_info_th = extract_informal(os.path.join(info_directory, "valid"))

    # Store the lean and isabelle version for each theorem
    valid_theorems = {}
    for (lean_name, lean_th), (isab_name, isab_th), (info_name, info_th) in zip(valid_lean_th, valid_isab_th, valid_info_th):
        if lean_name != isab_name or isab_name != info_name:
            print("Error: the Lean, Isabelle and informal description databases contain different theorems.")
            sys.exit(1)
        else:
            valid_theorems[isab_name] = {
                "lean": lean_th,
                "isabelle": isab_th,
                "informal": info_th
            }

    # The same is done for test theorems
    test_lean_th = extract_lean(os.path.join(lean_directory, "test.lean"))
    test_isab_th = extract_isabelle(os.path.join(isab_directory, "test"))
    test_info_th = extract_informal(os.path.join(info_directory, "test"))

    test_theorems = {}
    for (lean_name, lean_th), (isab_name, isab_th), (info_name, info_th) in zip(test_lean_th, test_isab_th, test_info_th):
        if lean_name != isab_name or isab_name != info_name:
            print("Error: the Lean, Isabelle and informal description databases contain different theorems.")
            sys.exit(1)
        else:
            test_theorems[isab_name] = {
                "lean": lean_th,
                "isabelle": isab_th,
                "informal": info_th
            }

    # Return the theorems
    return {"valid": valid_theorems, "test": test_theorems}