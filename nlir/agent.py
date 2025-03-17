import os
import json
import random
import string
import re
from abc import ABC, abstractmethod
from typing import Iterable
from pathlib import Path
from omegaconf import DictConfig
import concurrent.futures
import weave
from weave.trace.util import ContextAwareThreadPoolExecutor
from litellm import completion
from litellm.exceptions import UnsupportedParamsError, BadRequestError
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

    def log(self, message: Message | Response):
        with open(self.log_file, "a") as file:
            match message:
                case Response():
                    print(message.model_dump_json(), file=file)
                case _:
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

    def multi_responses_seed(self, messages: list[Message], n=1) -> list[Response]:
        """
        The diversity of the responses is obtained by adding different seeds when asking to the LLM.
        """
        list(map(self.log, messages))
        seeds = [random.randint(0, 9999) for _ in range(n)]

        def seeded_response(seed):
            raw = completion(
                model=self.model_id,
                messages=messages,
                seed=seed,
                temperature=self.temperature,
            )
            resp = Response(role="assistant", content=raw.choices[0].message.content)
            self.log(resp)
            return resp

        return list(map(seeded_response, seeds))

    def multi_responses_prefix(self, messages: list[Message], n=1, k=100) -> list[Response]:
        """
        The diversity of the responses is obtained by adding random characters at the beginning of the prompt.
        """
        list(map(self.log, messages))
        prefixes = [''.join(random.choices(string.ascii_uppercase + string.digits, k=k)) for _ in range(n)]

        def prefixed_response(prefix):
            last_message = messages[-1].copy()
            last_message["content"] = prefix + "\n" + last_message["content"] # essayer de mettre la prefix à la fin
            raw = completion(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
            )
            resp = Response(role="assistant", content=raw.choices[0].message.content)
            self.log(resp)
            return resp

        return list(map(prefixed_response, prefixes))

    def multi_responses_personality(self, messages: list[Message], n=1) -> list[Response]:
        """
        The diversity of the responses is obtained by giving a last instruction on how the LLM should behave.
        """

        prompt = "One last instruction, while you try to give the best answer possible you should be very {personality}."

        personalities = [
            "calm",
            "cautious",
            "exhaustive",
            "thorough",
            "succint",
            "careful",
            "conscientious",
            "couragous",
            "bold",
            "positive",
            "negative",
            "excited",
            "efficient",
            "scientifically accurate",
            "precise",
            "pernickety",
            "finical",
            "picky",
            "niggling",
            "detailed",
            "pathetic",
            "sensitive",
        ]
        personalities = random.sample(personalities, n)

        def personality_response(personality):
            print(personality)
            last_message = { "role" : "user", "content" : prompt.format(personality=personality) }
            raw = completion(
                model=self.model_id,
                messages=messages+[last_message],
                temperature=self.temperature,
            )
            resp = Response(role="assistant", content=raw.choices[0].message.content)
            self.log(resp)
            return resp

        return list(map(personality_response, personalities))

    @weave.op()
    def multi_responses(self, messages: list[Message], n=1, mode=1) -> list[Response]:
        """
        Given a list of messages returns n possible responses.
        """
        match mode:
            case 0:
                return self.multi_responses_seed(messages, n)
            case 1:
                return self.multi_responses_prefix(messages, n)
            case 2:
                return self.multi_responses_personality(messages, n)
            case _:
                raise Exception("Wrong multi_responses mode.")


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
    def response(self, messages: list[Message], seed: int = 1234) -> Response:
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
