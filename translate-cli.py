import json
import hydra
from omegaconf import DictConfig
from datetime import datetime
from pathlib import Path
from pytanque import Pytanque
import sys
from tqdm import tqdm
from functools import partial

from nlir.agent import GPT
from nlir.petanque import TranslateEnv
from nlir.search import naive_search, beam_search
from nlir.translate.prompt import Prompt, prompt_list
from nlir.translate.extract import extract_coq_theorem, extract_theorems

def decode_response(file_path: str):
    messages = []
    with open(file_path, 'r') as file:
        for line in file:
            messages.append(json.loads(line))

    return messages[-1]["content"]

def translate_one_thm(res_path: str, log_path: str, cfg_translator: DictConfig, theorem, prompt: Prompt):
    """
    Translates a theorem to Coq and write it in the appropriate Coq file.

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
def translate_once(cfg: DictConfig):
    """
    Translates one or a complete set of theorems to Coq in only one try.

    Args:
        cfg (Dictconfig): The configuration needed for the translation.

    Returns:
        Nothing, translates theorems in resulting files
    """

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
        translate_one_thm(str(res_path), str(log_path), cfg.agent, theorem, prompt)
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
        translate_one_thm(str(res_path), str(log_path), cfg.agent, theorem, prompt)
    print("Done translating valid theorems.")

    print("Translating test theorems.")
    for theorem in tqdm(theorems["test"].items()):
        res_path = Path(cfg.workspace + f"/coq/test/{cfg.prompt}_{dt}", f"{theorem[0]}.v").absolute()
        res_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = Path(cfg.log_dir + f"/test/{cfg.prompt}_{dt}", f"{theorem[0]}.jsonl").absolute()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        translate_one_thm(str(res_path), str(log_path), cfg.agent, theorem, prompt)
    print("Done translating test theorems.")

    sys.exit(0)

@hydra.main(version_base=None, config_path="conf", config_name="translate_config")
def translate(cfg: DictConfig):
    """
    Translates one or a complete set of theorems to Coq in several try.

    Args:
        cfg (Dictconfig): The configuration needed for the translation.

    Returns:
        Nothing, translates theorems in resulting files
    """

    # Set up the petanque client
    wk_path = Path(cfg.workspace).expanduser().absolute()
    pet = Pytanque(cfg.petanque.address, cfg.petanque.port)
    pet.connect()
    pet.set_workspace(False, str(wk_path))

    # Set the search function
    match cfg.search.mode:
        case "naive":
            search = naive_search
        case "beam":
            search = partial(
                beam_search,
                beam_size=cfg.search.beam_size,
                n_reponses=cfg.search.n_responses,
            )
        case _:
            raise RuntimeError("search.mode config should be one of [naive, beam]")

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

        # Set up the log file and the result file
        dt = datetime.now().strftime("%y%m%d_%H%M%S")
        log_path = Path(cfg.log_dir + "/single", f"{cfg.theorem}_{cfg.prompt}_{dt}.jsonl").absolute()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_path = Path(cfg.workspace + "/coq/single", f"{cfg.theorem}_{cfg.prompt}_{dt}.v").absolute()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create the agent
        agent = GPT(str(log_path), cfg.agent)

        # Create the petanque env
        env = TranslateEnv(pet, str(wk_path), str(file_path), cfg.theorem, cfg.petanque.context)

        # Start translating
        print(f"Try to prove {cfg.theorem} in {file_path}.")
        status = search(agent, env, cfg.search.max_steps)
        print(f"\n\n--- Success: {status.success} ---")
        print("---\n\n")

        sys.exit(0)

if __name__ == "__main__":
    translate()