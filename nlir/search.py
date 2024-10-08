from .petanque import Env
from .agent import LLM
from dataclasses import dataclass
from typing import List


@dataclass
class Status:
    steps: int
    success: bool


def naive_search(agent: LLM, env: Env, max_steps: int) -> Status:
    response = agent.response(env.prompt)  # Initial step
    for step in range(max_steps):
        env.exec(response)
        print(env.proof)
        if env.proof_finished:
            proof = " ".join(env.proof)
            agent.log({"role": "user", "content": f"Final Proof: {proof}"})
            return Status(step, True)
        else:
            if len(str(env.prompt)) > 100000:
                # prompt is too big!
                break
            response = agent.response(env.prompt)

    if not env.proof_finished:
        proof = " ".join(env.proof)
        agent.log({"role": "user", "content": f"Partial Proof: {proof}"})
        return Status(max_steps, False)

    raise RuntimeError("Unreachable code.")
