import click
import sys
from pytanque import Pytanque
from nlir.agent import LLM, Ghost, GPT
from nlir.petanque import Env, TacticEnv, TemplateEnv
from nlir.search import naive_search, Status
from pathlib import Path
from hydra import compose, initialize
from datetime import datetime
from omegaconf import DictConfig
from typing import Callable, Type


def load_conf(
    conf_file: str,
) -> tuple[DictConfig, Pytanque, Path, Type[Env], Callable[[LLM, Env, int], Status]]:

    # Load config
    cfg_path = Path("conf")
    if not cfg_path.exists():
        print(f"Config files should be in the conf directory.", file=sys.stderr)
        sys.exit(1)
    with initialize(version_base=None, config_path=str(cfg_path)):
        cfg = compose(config_name=conf_file)

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

    return cfg, pet, wk_path, env_cls, search


@click.group()
def cli():
    pass


@cli.command()
@click.option("-f", "--file", help="Coq file", required=True)
@click.option("-t", "--thm", help="theorem name", required=True)
@click.option("-l", "--log-file", help="conversation log", required=True)
@click.option("-c", "--conf", help="conf directory", default="config")
def replay(file: str, thm: str, log_file, conf: str):
    """
    Use a log file to replay a conversation
    """

    cfg, pet, wk_path, env_cls, search = load_conf(conf)

    source_path = Path(log_file)
    if not source_path.exists():
        print(f"Log file {log_file} not found", file=sys.stderr)
        sys.exit(1)

    file_path = Path(wk_path, file)

    print(f"Replay the proof of {thm} from {file}")
    env = env_cls(pet, str(wk_path), str(file_path), thm, cfg.petanque.context)
    agent = Ghost(source_path.resolve())
    search(agent, env, cfg.search.max_steps)


@cli.command()
@click.option("-f", "--file", help="Coq file", required=True)
@click.option("-t", "--thm", help="theorem name", required=True)
@click.option("-c", "--conf", help="conf directory", default="conf")
def prove(file: str, thm: str, conf: str):
    """
    Use the configs and logs in log_dir to replay the proof.
    """

    cfg, pet, wk_path, env_cls, search = load_conf(conf)

    file_path = Path(wk_path, file)
    dt = datetime.now().strftime("%y%m%d-%H%M%S")
    log_file = f"{file}:{thm}_{dt}.jsonl"

    print(f"Try to prove {thm} from {file}")
    env = env_cls(pet, str(wk_path), str(file_path), thm, cfg.petanque.context)
    agent = agent = GPT(
        log_file,
        cfg.agent.model_id,
        cfg.agent.temperature,
    )
    search(agent, env, cfg.search.max_steps)


if __name__ == "__main__":
    cli()
