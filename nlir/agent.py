import os
import openai as oai
import json
from abc import ABC, abstractmethod
from typing import Iterable
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessage
from datetime import datetime
from pathlib import Path


class LLM(ABC):
    """
    Base class for LLM agents.
    `response` and `multi_responses` are used in `search`.
    """

    def __init__(self, log_file: str):
        self.log_file = log_file

    def log(self, message: ChatCompletionMessageParam):
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
        Given an list of messages returns the Agent response.
        """
        pass

    @abstractmethod
    def multi_responses(
        self, messages: Iterable[ChatCompletionMessageParam], n: int
    ) -> list[ChatCompletionMessage]:
        """
        Given an list of messages returns n possible responses.
        """
        pass


class GPT(LLM):
    """
    GPT based agent (OpenAI API)
    """

    def __init__(
        self,
        log_file: str,
        model_id: str,
        temperature: float,
    ):

        super().__init__(log_file)
        self.model_id = model_id
        self.temperature = temperature
        self.client = oai.OpenAI(
            project=os.environ["OPENAI_PROJECT"],
            api_key=os.environ["OPENAI_API_KEY"],
        )

    def response(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> ChatCompletionMessage:
        list(map(self.log, messages))
        resp = (
            self.client.chat.completions.create(
                model=self.model_id, messages=messages, temperature=self.temperature
            )
            .choices[0]
            .message
        )
        self.log(resp)
        return resp

    def multi_responses(
        self, messages: Iterable[ChatCompletionMessageParam], n=1
    ) -> list[ChatCompletionMessage]:
        resp = self.client.chat.completions.create(
            model=self.model_id, messages=messages, temperature=self.temperature, n=n
        )
        return [c.message for c in resp.choices]


class Ghost(LLM):
    """
    Ghost agent to replay a conversation from a log file `log_file.jsonl`.
    New conversation will be stored in `orig_log_file_replay.jsonl`.
    """

    def __init__(self, source_file):
        source = Path(source_file)
        super().__init__(os.path.join(source.parent, f"{source.stem}_replay.jsonl"))
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
        raise NotImplementedError
