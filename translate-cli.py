import json
import hydra
from omegaconf import DictConfig
from datetime import datetime
from pathlib import Path
import sys
from tqdm import tqdm

from nlir.agent import GPT
from nlir.translate.prompt import Prompt, prompt_list
from nlir.translate.extract_theorems import extract_all

def decode_response(file_path: str):
    messages = []
    with open(file_path, 'r') as file:
        for line in file:
            messages.append(json.loads(line))

    return messages[-1]["content"]

def translate(file_path: str, cfg_translator: DictConfig, theorem, prompt: Prompt):
    agent = GPT(file_path, cfg_translator)
    prompt = prompt.make(theorem)
    return agent.response(prompt)

@hydra.main(version_base=None, config_path="conf", config_name="translate_config")
def main(cfg: DictConfig):

    # Extract all theorems
    theorems = extract_all(cfg.workspace)

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

        dt = datetime.now().strftime("%y%m%d-%H%M%S")
        res_path = Path(cfg.output_dir + "/thm", f"{cfg.theorem}:{cfg.prompt}_{dt}.jsonl").absolute()
        res_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Try to translate {cfg.theorem}.")
        translate(str(res_path), cfg.agent, theorem, prompt)
        print("Done.")

        print(decode_response(res_path))

        sys.exit(0)

    # Translates all theorems
    dt = datetime.now().strftime("%y%m%d-%H%M%S")

    print("Translating valid theorems.")
    for theorem in tqdm(theorems["valid"].items()):
        res_path = Path(cfg.output_dir + f"/{cfg.prompt}_{dt}/valid", f"{theorem[0]}.jsonl").absolute()
        res_path.parent.mkdir(parents=True, exist_ok=True)
        translate(str(res_path), cfg.agent, theorem, prompt)
    print("Done translating valid theorems.")

    print("Translating test theorems.")
    for theorem in tqdm(theorems["test"].items()):
        res_path = Path(cfg.output_dir + f"/{cfg.prompt}_{dt}/test", f"{theorem[0]}.jsonl").absolute()
        res_path.parent.mkdir(parents=True, exist_ok=True)
        translate(str(res_path), cfg.agent, theorem, prompt)
    print("Done translating test theorems.")

    sys.exit(0)

if __name__ == "__main__":
    main()