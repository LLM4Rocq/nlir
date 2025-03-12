from .petanque import Env, TemplateEnv
from .agent import LLM, SystemMessage, UserMessage, Message, Response
from dataclasses import dataclass
from typing import List
import re
from typing import Iterable
from .prompts import comparison_prompts, tactic_prompts, template_prompts
from functools import partial
from ast import literal_eval
import weave
import heapq


@dataclass
class Status:
    steps: int
    success: bool
    proof: str


def naive_name(call):
    return f"Naive-{call.attributes['kind']}-{call.attributes['thm']}-{call.attributes['file']}"


@weave.op(call_display_name=naive_name)
def naive_search(agent: LLM, env: Env, max_steps: int) -> Status:
    for step in range(max_steps):
        response = agent.response(env.prompt)
        env.exec(response)
        proof = " ".join(env.proof)
        print(f"\nNaive search, step {step}:\n{proof}")
        if env.proof_finished:
            if env.check_proof():
                agent.log(UserMessage(role="user", content=f"Final Proof: {proof}"))
                return Status(step, True, proof)
            else:
                agent.log(UserMessage(role="user", content=f"Failed Proof: {proof}"))
                return Status(step, False, proof)
        else:
            env_prompt = env.prompt
            if len(str(env_prompt)) > 100000:
                # prompt is too big!
                break
            elif isinstance(env, TemplateEnv):
                if len(env.holes) > max_steps - 1 - step:
                    # number of remaining steps is too big
                    break

    if not env.proof_finished:
        proof = " ".join(env.proof)
        agent.log(UserMessage(role="user", content=f"Partial Proof: {proof}"))
        return Status(max_steps, False, proof)

    raise RuntimeError("Unreachable code.")


def create_comparison_prompt(
    list_env: list[Env],
) -> List[Message]:
    """
    Build the comparison prompt from the list of environments.
    """
    if list_env[0].__class__.__name__ == "TacticEnv":
        system_prompt = tactic_prompts.comparison_system_prompt
    else:
        system_prompt = template_prompts.comparison_system_prompt
    intro = comparison_prompts.user_prompt.format(
        theorem_code=list_env[0].thm_code,
    )
    attempts = "\n\n".join(
        [
            f"- Attempt {i}:\n\n{env.info_for_comparison}"
            for i, env in enumerate(list_env)
        ]
    )
    content = "\n\n".join([intro, attempts])
    return [
        SystemMessage(role="system", content=system_prompt),
        UserMessage(role="user", content=content),
    ]


@weave.op()
def parse_comparison(response: Response) -> List[int]:
    if response.content:
        match = re.match(
            r".*(\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)?\]).*", response.content, re.DOTALL
        )
        if match:
            try:
                return literal_eval(match[1])
            except ValueError:
                pass
        # model didn't format the list properly, try alternative way of parsing
        # simply get all numbers followed by a comma, end of string or newline
        # TBC
        if "Response" in response.content:
            to_parse = response.content.split("Response")[-1]
        else:
            to_parse = response.content
        match = re.findall(r"([0-9]+)(,|$|\n)", to_parse, re.DOTALL)
        parsed = [int(el[0]) for el in match]
        return parsed
    return []


@weave.op()
def sort_LLM(new_beam: list[Env], agent: LLM) -> list[Env]:
    comparison_prompt = create_comparison_prompt(new_beam)
    response = agent.response(comparison_prompt)
    perm_indices = parse_comparison(response)
    beam_size = len(new_beam)
    if len(perm_indices) < beam_size or not all([i < beam_size for i in perm_indices]):
        perm_indices = list(range(beam_size))
    return [new_beam[i] for i in perm_indices]


def sort_holes(new_beam: list[TemplateEnv]) -> list[TemplateEnv]:
    return sorted(new_beam, key=lambda x: len(x.holes))


@weave.op()
def expand_beam(
    beam: list[Env],
    agent: LLM,
    n_reponses: int,
    remaining_steps: int,
) -> list[Env]:
    new_beam = []
    for env in beam:
        responses = agent.multi_responses(env.prompt, n_reponses)
        for response in responses:
            env_copy = env.deepcopy()
            env_copy.exec(response)
            proof = " ".join(env_copy.proof)
            print(proof)
            if env_copy.proof_finished:
                if env_copy.check_proof():
                    agent.log(UserMessage(role="user", content=f"Final Proof: {proof}"))
                    return [env_copy]
            else:
                try:
                    if len(str(env_copy.prompt)) > 100000:
                        # prompt is too big, env will not be added to the beam
                        continue
                    elif env_copy.proof in [e.proof for e in new_beam]:
                        # avoid adding duplicate proofs
                        continue
                    elif isinstance(env_copy, TemplateEnv):
                        if len(env_copy.holes) > remaining_steps:
                            # avoid adding proofs with too many holes
                            continue
                except RuntimeError:
                    continue
            new_beam.append(env_copy)
    return new_beam


def bs_name(call):
    return f"BS-{call.attributes['kind']}-{call.attributes['thm']}-{call.attributes['file']}"


@weave.op(call_display_name=bs_name)
def beam_search(
    agent: LLM,
    env: Env,
    max_steps: int,
    beam_size: int,
    n_reponses: int,
    sorting_holes=False,
) -> Status:
    sort_beam = partial(sort_LLM, agent=agent)
    if sorting_holes:
        if isinstance(env, TemplateEnv):
            sort_beam = sort_holes
        else:
            raise RuntimeError("Sorting holes is only available for TemplateEnv")
    beam = [env]
    for step in range(max_steps):
        print(f"\nBeam search, step {step}:\n")
        # expand bean
        new_beam = expand_beam(beam, agent, n_reponses, max_steps - step)
        if new_beam:
            if len(new_beam) <= beam_size:
                if new_beam[0].proof_finished:
                    proof = " ".join(new_beam[0].proof)
                    return Status(step, True, proof)
                else:
                    beam = new_beam
            else:
                # sort new_beam
                beam = sort_beam(new_beam)
                beam = beam[:beam_size]
                print("sorted beam:")
                for b in beam:
                    print(" ".join(b.proof))
        else:
            proof = " ".join(beam[0].proof)
            agent.log(
                UserMessage(role="user", content=f"Failed Proof (empty beam): {proof}")
            )
            return Status(step, False, proof)

    # beam = sort_holes(beam)
    proof = " ".join(beam[0].proof)
    agent.log(UserMessage(role="user", content=f"Partial Proof: {proof}"))
    return Status(max_steps, False, proof)


def best_first_search(
    agent: LLM, env: Env, max_steps: int, n_reponses: int, max_tokens: int
) -> Status:
    queue = [(0.0, env)]
    for step in range(max_steps):
        if len(queue) == 0:
            break
        total_score, current_env = heapq.heappop(queue)

        step_cands, step_scores = agent.generate(
            num_sampples=n_reponses,
            max_tokens=max_tokens,
            prompt=current_env.prompt,
            stop="---",
        )
        step_cands = [s.strip() for s in step_cands]

        for step_cand, score in zip(step_cands, step_scores):
            env_copy = current_env.deepcopy()
            env_copy.exec([step_cand])
            proof = " ".join(env_copy.proof)
            print(proof)
            if env_copy.proof_finished:
                if env_copy.check_proof():
                    agent.log(UserMessage(role="user", content=f"Final Proof: {proof}"))
                    return Status(step, True, proof)
            else:
                # Score is negative log probability summed across steps
                new_score = total_score - score
                heapq.heappush(queue, (new_score, env_copy))

    agent.log(UserMessage(role="user", content=f"Failed Proof: {proof}"))
    return Status(step, False, proof)
