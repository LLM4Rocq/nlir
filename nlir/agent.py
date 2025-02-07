import os
import json
from abc import ABC, abstractmethod
from typing import Iterable
from pathlib import Path
from omegaconf import DictConfig
import concurrent.futures
import weave
from weave.trace.util import ContextAwareThreadPoolExecutor
from litellm import completion
from litellm.exceptions import UnsupportedParamsError
from openai.types.chat import (
    ChatCompletionSystemMessageParam as SystemMessage,
    ChatCompletionUserMessageParam as UserMessage,
    ChatCompletionMessageParam as Message,
    ChatCompletionMessage as Response,
)


class LLM(ABC):
    """
    Base class for LLM agents.
    `response` and `multi_responses` are used in `search`.
    """

    def __init__(self, log_file: str):
        self.log_file = log_file
        # Delete the log file if it exists
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        # Delete the log file if it exists
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def log(self, message: Message | Response):
        with open(self.log_file, "a") as file:
            match message:
                case Response():
                    print(message.model_dump_json(), file=file)
                case _:
                    print(json.dumps(message, ensure_ascii=False), file=file)
                    print(json.dumps(message, ensure_ascii=False), file=file)

    @abstractmethod
    @weave.op()
    def response(self, messages: list[Message]) -> Response:
        """
        Given a list of messages returns the Agent response.
        """
        pass

    @abstractmethod
    @weave.op()
    def multi_responses(self, messages: list[Message], n: int) -> list[Response]:
        """
        Given a list of messages returns n possible responses.
        """
        pass


class LiteLLM(LLM):
    """
    LiteLLM based agent (OpenAI API)
    """

    def __init__(
        self,
        log_file: str,
        cfg_agent: DictConfig,
    ):

        super().__init__(log_file)
        self.model_id = cfg_agent.model_id
        self.temperature = cfg_agent.temperature

    @weave.op()
    def response(self, messages: list[Message]) -> Response:
        list(map(self.log, messages))
        raw = completion(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
        )
        # Hack: surprising type for raw.choices...
        resp = Response(role="assistant", content=raw.choices[0].message.content)  # type: ignore
        self.log(resp)
        return resp

    @weave.op()
    def multi_responses(self, messages: list[Message], n=1) -> list[Response]:
        list(map(self.log, messages))
        try:
            raw = completion(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                n=n,
            )
            resp = [
                # Hack: surprising type for raw.choices...
                Response(role="assistant", content=m.message.content)  # pyright: ignore
                for m in raw.choices  # type: ignore
            ]
        except UnsupportedParamsError:
            with ContextAwareThreadPoolExecutor(max_workers=20) as executor:
                these_futures = [
                    executor.submit(self.response, messages) for _ in range(n)
                ]
                concurrent.futures.wait(these_futures)
                resp = [future.result() for future in these_futures]
        list(map(self.log, resp))
        return resp


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
                line = line.encode("utf-8")
                logs.append(json.loads(line))
        self.messages = filter(lambda m: m["role"] == "assistant", logs)

    def __iter__(self) -> Iterable[Response]:
        yield from self.messages

    @weave.op()
    def response(self, messages: list[Message]) -> Response:
        list(map(self.log, messages))
        resp = next(self.messages)
        self.log(resp)
        return Response(**resp)

    @weave.op()
    def multi_responses(self, messages: list[Message], n=1) -> list[Response]:
        list(map(self.log, messages))
        # resp = [next(self.messages) for i in range(n)]
        resp = []
        for i in range(n):
            try:
                resp.append(next(self.messages))
            except StopIteration:
                break
        resp = [Response(**r) for r in resp]
        list(map(self.log, resp))
        return resp
