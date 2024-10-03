import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytanque import Pytanque
from nlir.agent import GPT
from nlir.petanque import TacticEnv, TemplateEnv
from nlir.search import naive_search


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run(cfg: DictConfig) -> None:

    pet = Pytanque(cfg.petanque.address, cfg.petanque.port)
    pet.connect()
    pet.set_workspace(False, cfg.workspace)

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

    for thms in cfg.benchmark:
        print(f"Inspecting {thms.file}")
        for thm in thms.theorems:
            print(f"  Try {thm}")
            env = TemplateEnv(pet, cfg.workspace, thms.file, thm)
            log_file = os.path.join(log_dir, f"{thms.file}:{thm}")
            agent = GPT(
                log_file,
                cfg.agent.model_id,
                cfg.agent.temperature,
            )
            search(agent, env, cfg.search.max_steps)


if __name__ == "__main__":
    run()
