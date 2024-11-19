import os
import openai as oai
import json
from abc import ABC, abstractmethod
from typing import Iterable
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessage
from pathlib import Path
from omegaconf import DictConfig
import concurrent.futures
import weave

class LLM(ABC):
    """
    Base class for LLM agents.
    `response` and `multi_responses` are used in `search`.
    """

    def __init__(self, log_file: str, cfg_agent: DictConfig):
        self.log_file = log_file
        self.cfg_agent = cfg_agent

    def log(self, message: ChatCompletionMessageParam | ChatCompletionMessage):
        with open(self.log_file, "a") as file:
            match message:
                case ChatCompletionMessage():
                    print(message.model_dump_json(), file=file)
                case _:
                    print(json.dumps(message), file=file)

    @abstractmethod
    def response(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> ChatCompletionMessage:
        """
        Given a list of messages returns the Agent response.
        """
        pass

    @abstractmethod
    def multi_responses(
        self, messages: Iterable[ChatCompletionMessageParam], n: int
    ) -> list[ChatCompletionMessage]:
        """
        Given a list of messages returns n possible responses.
        """
        pass


class GPT(LLM):
    """
    GPT based agent (OpenAI API)
    """

    def __init__(
        self,
        log_file: str,
        cfg_agent: DictConfig,
    ):

        super().__init__(log_file, cfg_agent)
        #self.model_id = model_id
        #self.temperature = temperature
        if self.cfg_agent.local:
            self.client = oai.OpenAI(
                api_key="EMPTY",
                base_url="http://localhost:8000/v1",
            )
            self.chat_complete = self.client.chat.completions.create
        else:
            if self.cfg_agent.provider == "openai":
                self.client = oai.OpenAI(
                    project=os.environ["OPENAI_PROJECT"],
                    api_key=os.environ["OPENAI_API_KEY"],
                )
                self.chat_complete = self.client.chat.completions.create
            elif self.cfg_agent.provider == "deepseek": 
                self.client = oai.OpenAI(
                    api_key=os.environ["DEEPSEEK_API_KEY"], 
                    base_url="https://api.deepseek.com"
                )
                self.chat_complete = self.client.chat.completions.create
            elif self.cfg_agent.provider == "mistral":
                from mistralai import Mistral
                if self.cfg_agent.model_id.startswith("codestral"):
                    self.client = Mistral(api_key=os.environ["CODESTRAL_API_KEY"], server_url='https://codestral.mistral.ai')
                else:
                    self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"]
                    )
                self.chat_complete = self.client.chat.complete
            else:
                raise RuntimeError("Unknown provider")

    @weave.op()
    def response(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> ChatCompletionMessage:
        list(map(self.log, messages))
        resp = (
            self.chat_complete(
                model=self.cfg_agent.model_id, messages=messages, temperature=self.cfg_agent.temperature
            )
            .choices[0]
            .message
        )
        if self.cfg_agent.provider == "mistral":
            resp = ChatCompletionMessage(**resp.dict())    
        self.log(resp)
        return resp

    @weave.op()
    def multi_responses(
        self, messages: Iterable[ChatCompletionMessageParam], n=1
    ) -> list[ChatCompletionMessage]:
        list(map(self.log, messages))
        if self.cfg_agent.provider == "deepseek":
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                these_futures = [executor.submit(self.response, messages) for _ in range(n)]
                concurrent.futures.wait(these_futures)
                resp = [future.result() for future in these_futures]

        else:
            resp = self.chat_complete(
            model=self.cfg_agent.model_id, messages=messages, temperature=self.cfg_agent.temperature, n=n
        )
        if self.cfg_agent.provider == "mistral":
            for c in resp.choices:
                self.log(c.message.dict())
            return [ChatCompletionMessage(**c.message.dict()) for c in resp.choices]
        elif self.cfg_agent.provider == "deepseek":
            return resp
        else:
            for c in resp.choices:
                self.log(c.message)
            return [c.message for c in resp.choices]


class Ghost(LLM):
    """
    Ghost agent to replay a conversation from a log file `log_file.jsonl`.
    New conversation will be stored in `orig_log_file_replay.jsonl`.
    """

    def __init__(self, source_file):
        source = Path(source_file)
        super().__init__(os.path.join(source.parent, f"{source.stem}_replay.jsonl"), cfg_agent=None)
        logs = []
        with open(source_file, "r") as file:
            for line in file:
                logs.append(json.loads(line))
        self.messages = filter(lambda m: m["role"] == "assistant", logs)

    def __iter__(self) -> Iterable[ChatCompletionMessage]:
        yield from self.messages

    def response(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> ChatCompletionMessage:
        list(map(self.log, messages))
        resp = next(self.messages)
        self.log(resp)
        return ChatCompletionMessage(**resp)

    def multi_responses(
        self, messages: Iterable[ChatCompletionMessageParam], n=1
    ) -> list[ChatCompletionMessage]:
        list(map(self.log, messages))
        resp = [next(self.messages) for i in range(n)]
        for r in resp:
            self.log(r)
        return [ChatCompletionMessage(**r) for r in resp]
