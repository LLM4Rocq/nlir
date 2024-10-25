from .petanque import Env
from .agent import LLM
from dataclasses import dataclass
from typing import List
import re
from typing import Iterable
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessage
from .prompts import comparison_prompts


@dataclass
class Status:
    steps: int
    success: bool
    proof: str


def naive_search(agent: LLM, env: Env, max_steps: int) -> Status:
    response = agent.response(env.prompt)  # Initial step
    for step in range(max_steps):
        env.exec(response)
        print(env.proof)
        if env.proof_finished:
            proof = " ".join(env.proof)
            if env.check_proof():
                agent.log({"role": "user", "content": f"Final Proof: {proof}"})
                return Status(step, True, proof)
            else:
                agent.log({"role": "user", "content": f"Failed Proof: {proof}"})
                return Status(step, False, proof)
        else:
            if len(str(env.prompt)) > 100000:
                # prompt is too big!
                break
            response = agent.response(env.prompt)

    if not env.proof_finished:
        proof = " ".join(env.proof)
        agent.log({"role": "user", "content": f"Partial Proof: {proof}"})
        return Status(max_steps, False, proof)

    raise RuntimeError("Unreachable code.")


def create_comparison_prompt(list_env: list[Env]) -> Iterable[ChatCompletionMessageParam]:
    """
    Build the comparison prompt from the list of environments.
    """
    system_prompt = comparison_prompts.comparison_system_prompt
    intro = comparison_prompts.user_prompt.format(
        theorem_code=list_env[0].thm_code,
    )
    attempts = "\n\n".join([f"Here is the {i}th attempt:\n\n{env.prompt_for_comparison}" 
                                 for i, env in enumerate(list_env)])
    content = "\n\n".join([intro, attempts])    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def parse_comparison(message: ChatCompletionMessage, expanded_beam_size: int) -> List[int]:
    try:
        parsed = re.match(r'.*(\[\s*(\d+\s*(,\s*\d+\s*)*)?\]).*', message.content, re.DOTALL)
        if parsed is None or len(parsed) < 2:
            # model didn't format the list properly, try alternative way of parsing
            # simply get all numbers followed by a comma, end of string or newline
            # TBC
            if "Response" in message.content:
                to_parse = message.content.split("Response")[-1]
            else:
                to_parse = message.content
            parsed = re.findall(r'([0-9]+)(,|$|\n)', to_parse, re.DOTALL)
            parsed = [int(el[0]) for el in parsed]
            parsed = parsed[-expanded_beam_size:]  # remove potentially matched from reasoning blabla
        else:
            parsed = eval(parsed[1])
    except:
        parsed = list(range(expanded_beam_size))
    return parsed


def beam_search(agent: LLM, env: Env, max_steps: int, beam_size: int, n_reponses: int) -> Status:
    beam = [env]
    for step in range(max_steps):
        # expand bean
        new_beam = []
        for env in beam:
            responses = agent.multi_responses(env.prompt, n_reponses)
            for response in responses:
                env_copy = env.deepcopy()
                env_copy.exec(response)
                print(env_copy.proof)
                if env_copy.proof_finished:
                    proof = " ".join(env_copy.proof)
                    if env_copy.check_proof():
                        agent.log({"role": "user", "content": f"Final Proof: {proof}"})
                        return Status(step, True, proof)
                else:
                    if len(str(env_copy.prompt)) > 100000:
                        # prompt is too big, env will not be added to the beam
                        continue
                    new_beam.append(env_copy)
        
        if step < max_steps-1:
            if new_beam:
            # sort new_bean
                comparison_prompt = create_comparison_prompt(new_beam)
                response = agent.response(comparison_prompt)
                perm_indices = parse_comparison(response, len(new_beam))
                beam = [new_beam[i] for i in perm_indices][:beam_size]
            else:
                raise RuntimeError("Empty beam.")
        else:
            proof = " ".join(beam[0].proof)
            agent.log({"role": "user", "content": f"Partial Proof: {proof}"})
            return Status(max_steps, False, proof)
