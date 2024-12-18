import json
import hydra
from omegaconf import DictConfig
from datetime import datetime
from pathlib import Path
import sys
from tqdm import tqdm

from nlir.agent import GPT
from nlir.translate.prompt import Prompt, prompt_list
from nlir.translate.extract import extract_coq_theorem, extract_theorems

def decode_response(file_path: str):
    messages = []
    with open(file_path, 'r') as file:
        for line in file:
            messages.append(json.loads(line))

    return messages[-1]["content"]

def translate(res_path: str, log_path: str, cfg_translator: DictConfig, theorem, prompt: Prompt):
    """
    Translates the theorem to Coq and write it in the appropriate Coq file.

    Args:
        res_path (str): Path to the Coq file to be written.
        log_path (str): Path to the log file of the agent.
        cfg_translator (DictConfig): Configurations for the agent.
        theorem: The theorem to translate.
        prompt (Prompt): The prompt to be used by the agent.

    Returns:
        Nothing, writes the result at `res_path` and the log at `log_path`.
    """

    # Create the agent
    agent = GPT(log_path, cfg_translator)

    # Make a prompt
    prompt = prompt.make(theorem)

    # Get the agent response
    message = agent.response(prompt)
    message = message.content

    # Extract the Coq theorem
    theorem = extract_coq_theorem(message)

    # Write in the appropriate file
    with open(res_path, 'w') as file:
        file.write(theorem + "\nProof.\nAdmitted.")

    return theorem

@hydra.main(version_base=None, config_path="conf", config_name="translate_config")
def main(cfg: DictConfig):

    # Extract all theorems
    theorems = extract_theorems(cfg.workspace)

    # Find the prompt
    prompt = prompt_list[cfg.prompt]

    # Only translate one theorem
    if cfg.theorem:

        # Find the theorem
        theorem = None
        for name, code in theorems["valid"].items():
            if cfg.theorem == name:
                theorem = name, code
        for name, code in theorems["test"].items():
            if cfg.theorem == name:
                theorem = name, code

        if theorem is None:
            print(f"The theorem {cfg.theorem} has not been found.")
            sys.exit(1)

        dt = datetime.now().strftime("%y%m%d_%H%M%S")
        res_path = Path(cfg.workspace + "/coq/single", f"{cfg.theorem}_{cfg.prompt}_{dt}.v").absolute()
        res_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = Path(cfg.log_dir + "/single", f"{cfg.theorem}_{cfg.prompt}_{dt}.jsonl").absolute()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Try to translate {cfg.theorem}.")
        translate(str(res_path), str(log_path), cfg.agent, theorem, prompt)
        print("Done.")

        print(decode_response(log_path))

        sys.exit(0)

    # Translates all theorems
    dt = datetime.now().strftime("%y%m%d_%H%M%S")

    print("Translating valid theorems.")
    for theorem in tqdm(theorems["valid"].items()):
        res_path = Path(cfg.workspace + f"/coq/valid/{cfg.prompt}_{dt}", f"{theorem[0]}.v").absolute()
        res_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = Path(cfg.log_dir + f"/valid/{cfg.prompt}_{dt}", f"{theorem[0]}.jsonl").absolute()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        translate(str(res_path), str(log_path), cfg.agent, theorem, prompt)
    print("Done translating valid theorems.")

    print("Translating test theorems.")
    for theorem in tqdm(theorems["test"].items()):
        res_path = Path(cfg.workspace + f"/coq/test/{cfg.prompt}_{dt}", f"{theorem[0]}.v").absolute()
        res_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = Path(cfg.log_dir + f"/test/{cfg.prompt}_{dt}", f"{theorem[0]}.jsonl").absolute()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        translate(str(res_path), str(log_path), cfg.agent, theorem, prompt)
    print("Done translating test theorems.")

    sys.exit(0)

if __name__ == "__main__":
    main()