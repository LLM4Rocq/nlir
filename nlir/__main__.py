import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytanque import Pytanque
from nlir.agent import Ghost, GPT
from nlir.petanque import TacticEnv, TemplateEnv
from nlir.search import naive_search


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run(cfg: DictConfig) -> None:
    match cfg.agent.kind:
        case "ghost":
            agent = Ghost(cfg.agent.source_file)
        case "gpt":
            log_dir = HydraConfig.get().runtime.output_dir
            log_file = os.path.join(log_dir, f"{cfg.file}:{cfg.thm}")
            agent = GPT(
                log_file,
                cfg.agent.model_id,
                cfg.agent.temperature,
            )
        case _:
            raise RuntimeError("agent.kind config should be one of [ghost, gpt]")

    pet = Pytanque(cfg.petanque.address, cfg.petanque.port)
    pet.connect()
    pet.set_workspace(False, cfg.workspace)

    match cfg.search.kind:
        case "tactics":
            env = TacticEnv(pet, cfg.workspace, cfg.file, cfg.thm)
        case "template":
            env = TemplateEnv(pet, cfg.workspace, cfg.file, cfg.thm)
        case _:
            raise RuntimeError(
                "search.kind config should be one of [tactics, template]"
            )

    match cfg.search.mode:
        case "naive":
            naive_search(agent, env, 15)
        case "beam":
            raise NotImplementedError
        case _:
            raise RuntimeError("search.mode config should be one of [naive, beam]")


if __name__ == "__main__":
    run()
