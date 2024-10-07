from pytanque import Pytanque
from nlir.agent import LLM, Ghost, GPT
from nlir.petanque import Env, TacticEnv, TemplateEnv
from nlir.search import naive_search, Status
from pathlib import Path
import hydra
from datetime import datetime
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Try to prove the theorem `cfg.thm` from file `cfg.file`.
    If option replay is set to a log file, replay the conversation from the logs
    """

    wk_path = Path(cfg.workspace).expanduser().absolute()
    pet = Pytanque(cfg.petanque.address, cfg.petanque.port)
    pet.connect()
    pet.set_workspace(False, str(wk_path))

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

    dt = datetime.now().strftime("%y%m%d-%H%M%S")
    log_file = f"{cfg.file}:{cfg.thm}_{dt}.jsonl"
    file_path = Path(wk_path, cfg.file)

    if hasattr(cfg, "replay"):
        source_path = Path(cfg.replay)
        agent = Ghost(source_path.resolve())
    else:
        agent = agent = GPT(
            log_file,
            cfg.agent.model_id,
            cfg.agent.temperature,
        )

    env = env_cls(pet, str(wk_path), str(file_path), cfg.thm, cfg.petanque.context)

    print(f"Try to prove {cfg.thm} from {cfg.file}")
    search(agent, env, cfg.search.max_steps)


if __name__ == "__main__":
    main()
