import os
import openai as oai
from omegaconf import DictConfig
from functools import partial


def get_agent(cfg_agent: DictConfig):
    if cfg_agent.local:
        api_key = "EMPTY"
        if cfg_agent.provider == "ollama":
            base_url = "http://localhost:11434/v1"
        else:
            # TODO name of gpu needs to be retrieved at allocation
            base_url = "http://localhost:8000/v1"
        client = oai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        if cfg_agent.provider == "mistral":
            from mistralai import Mistral

            if cfg_agent.model_id.startswith("codestral"):
                client = Mistral(
                    api_key=os.environ["CODESTRAL_API_KEY"],
                    server_url="https://codestral.mistral.ai",
                )
            else:
                client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
            return client.chat.complete
        elif cfg_agent.provider == "anthropic":
            import anthropic

            api_key = os.environ["ANTHROPIC_API_KEY"]
            client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
            return client.messages.create
        elif cfg_agent.provider == "openai":
            project = os.environ["OPENAI_PROJECT"]
            api_key = os.environ["OPENAI_API_KEY"]
            client = oai.OpenAI(project=project, api_key=api_key)
        else:
            if cfg_agent.provider == "xai":
                api_key = os.environ["XAI_API_KEY"]
                base_url = "https://api.x.ai/v1"
            elif cfg_agent.provider == "deepseek":
                api_key = os.environ["DEEPSEEK_API_KEY"]
                base_url = "https://api.deepseek.com"
            else:
                raise RuntimeError("Unknown provider")

            client = oai.OpenAI(api_key=api_key, base_url=base_url)

    return client.chat.completions.create


def allow_mutli_responses(provider):
    if provider in ["deepseek", "ollama", "anthropic"]:
        # TBC for anthropic (and for xai)
        return False
    else:
        return True
