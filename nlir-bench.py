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
from typing import Tuple


def check_config(
    pet: Pytanque, wk_path: Path, cfg: DictConfig
) -> list[Tuple[Path, str]]:
    errors = []
    theorems = []
    for thms in cfg.benchmark:
        file_path = Path(wk_path, thms.file).absolute()
        for thm in thms.theorems:
            try:
                pet.start(str(file_path), thm)
                theorems.append((file_path, thm))
            except PetanqueError as err:
                errors.append(f"- File {thms.file} {thm}: {err.message}")
    if not errors:
        print(f"Benchmarking {len(theorems)} theorems in {len(cfg.benchmark)} files")
    else:
        print(f"Config contains the following errors:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)
    return theorems


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig):
    wk_path = Path(cfg.workspace).expanduser().absolute()
    pet = Pytanque(cfg.petanque.address, cfg.petanque.port)
    pet.connect()
    pet.set_workspace(False, str(wk_path))

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
    theorems = check_config(pet, wk_path, cfg)

    for file_path, thm in theorems[: cfg.num_theorems]:
        print(f"Try to prove {thm} from {file_path.stem}")
        env = env_cls(pet, str(wk_path), str(file_path), thm, cfg.petanque.context)
        log_file = os.path.join(log_dir, f"{file_path.stem}:{thm}.jsonl")
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
