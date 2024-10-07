import os
import sys
import hydra
import json
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytanque import Pytanque, PetanqueError
from nlir.agent import GPT
from nlir.petanque import TacticEnv, TemplateEnv
from nlir.search import naive_search
from pathlib import Path


def check_config(pet: Pytanque, wk_path: Path, cfg: DictConfig):
    n_files = 0
    n_thms = 0
    errors = []
    for thms in cfg.benchmark:
        file_path = Path(wk_path, thms.file).absolute()
        n_files += 1
        for thm in thms.theorems:
            try:
                pet.start(str(file_path), thm)
                n_thms += 1
            except PetanqueError as err:
                errors.append(f"- File {thms.file} {thm}: {err.message}")
    if not errors:
        print(f"Benchmarking {n_thms} theorems in {n_files} files")
    else:
        print(f"Config contains the following errors:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig):

    wk_path = Path(cfg.workspace).expanduser().absolute()
    pet = Pytanque(cfg.petanque.address, cfg.petanque.port)
    pet.connect()
    pet.set_workspace(False, str(wk_path))

    check_config(pet, wk_path, cfg)

    log_dir = HydraConfig.get().runtime.output_dir

    match cfg.search.kind:
        case "tactics":
            env_cls = TacticEnv
        case "template":
            env_cls = TemplateEnv
        case _:
            raise RuntimeError(
                "search.kind config should be one of [tactics, template]"
            )

    match cfg.search.mode:
        case "naive":
            search = naive_search
        case "beam":
            raise NotImplementedError
        case _:
            raise RuntimeError("search.mode config should be one of [naive, beam]")

    results = {"names": [], "success": [], "steps": []}

    res_file = os.path.join(log_dir, "eval_results.json")

    for thms in cfg.benchmark:
        print(f"From {thms.file}")
        for thm in thms.theorems:
            print(f"  Try to prove {thm}")
            file_path = Path(wk_path, thms.file).absolute()
            env = env_cls(pet, str(wk_path), str(file_path), thm, cfg.petanque.context)
            log_file = os.path.join(log_dir, f"{thms.file}:{thm}.jsonl")
            agent = GPT(
                log_file,
                cfg.agent.model_id,
                cfg.agent.temperature,
            )
            status = search(agent, env, cfg.search.max_steps)
            results["names"].append(f"{env.file}:{env.thm}")
            results["success"].append(status.success)
            results["steps"].append(status.steps)
            with open(res_file, "w") as rf:
                json.dump(results, rf, indent=2)


if __name__ == "__main__":
    run()
