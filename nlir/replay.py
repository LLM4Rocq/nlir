import os
from pytanque import Pytanque
from nlir.agent import Ghost
from nlir.petanque import TacticEnv, TemplateEnv
from nlir.search import naive_search
import yaml


def replay(log_dir, file, thm):
    with open(os.path.join(log_dir, ".hydra", "config.yaml"), "r") as config_file:
        cfg = yaml.safe_load(config_file)
    pet = Pytanque(cfg["petanque"]["address"], cfg["petanque"]["port"])
    pet.connect()
    pet.set_workspace(False, cfg["workspace"])

    match cfg["search"]["kind"]:
        case "tactics":
            env_cls = TacticEnv
        case "template":
            env_cls = TemplateEnv
        case _:
            raise RuntimeError(
                "search.kind config should be one of [tactics, template]"
            )

    match cfg["search"]["mode"]:
        case "naive":
            search = naive_search
        case "beam":
            raise NotImplementedError
        case _:
            raise RuntimeError("search.mode config should be one of [naive, beam]")

    print(f"Replay {thm} from {file}")
    env = env_cls(pet, cfg["workspace"], file, thm)
    source_file = os.path.join(log_dir, f"{file}:{thm}.jsonl")
    agent = Ghost(source_file)
    search(agent, env, cfg["search"]["max_steps"])


if __name__ == "__main__":
    replay("outputs/2024-10-03/22-38-50", "foo.v", "foofoo")
