import re
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import deque
from typing import Iterable, Union, Tuple
from pytanque import Pytanque, State, Goal, PetanqueError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessage
from .prompts import tactic_prompts, template_prompts


def pp_goal(g: Goal) -> str:
    """
    Pretty print one goal.
    """
    hyps = "\n".join(
        [
            f"{', '.join(h.names)} {':= ' + h.def_ if h.def_ else ''} : {h.ty}"
            for h in g.hyps
        ]
    )
    return f"{hyps}\n⊢ {g.ty}"


def pp_goals(gs: list[Goal]) -> str:
    """
    Pretty print a list of goals.
    """
    return "\n".join(pp_goal(g) for g in gs)


def remove_comments(code):
    """
    Remove coq nested comments, e.g., (* this is a comment (* - induction n *) *).
    """

    # Remove code block delimiter
    code = code.strip()
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()

    pattern = re.compile(r"\(\*|\*\)")

    start = 0
    nested = 0
    indices_to_remove = []

    for match in pattern.finditer(code):
        if match.group() == "(*":
            nested += 1
            start = match.start()
        elif match.group() == "*)":
            nested -= 1
            if not nested:
                indices_to_remove.append((start, match.end()))

    cleaned_code = ""
    last_index = 0
    for start, end in indices_to_remove:
        cleaned_code += code[last_index:start]
        last_index = end
    cleaned_code += code[last_index:]

    return cleaned_code


def split_proof(
    proof: str,
    add_delimiter: bool = True,
) -> list[str]:
    """
    Split a proof in a list of tactics.
    """
    raw = remove_comments(proof)  # remove comments
    tactics = [
        t
        for b in re.split(r"(?<=\.)\s+(\++|\*+|\-+|{|})", raw)  # split proof in bullets
        for s in re.split(r"(?<=\.)\s+", b)  # split bullets in tactics
        if (t := s.strip())  # remove empty steps
    ]
    if add_delimiter:
        return ["{", *tactics, "}"]
    return tactics


class Env(ABC):
    """
    Base class for a Petanque environment.
    `exec` and `output` need to be defined for `search`
    """

    def __init__(self, pet, workspace, file, thm):
        self.pet = pet
        self.workspace = workspace
        self.file = file
        self.path = os.path.join(workspace, file)
        self.thm = thm
        self.proof: list[str] = []
        self.initial_state: State = self.pet.start(self.path, thm)
        self.thm_code = pp_goals(self.pet.goals(self.initial_state))
        self.n_interactions = 0

    @property
    @abstractmethod
    def proof_finished(self) -> bool:
        pass

    @abstractmethod
    def exec(self, message: ChatCompletionMessage):
        pass

    @property
    @abstractmethod
    def prompt(self) -> Iterable[ChatCompletionMessageParam]:
        pass


class TacticEnv(Env):
    """
    Tactic-by-tactic proof construction.
    """

    def __init__(self, pet: Pytanque, workspace: str, file: str, thm: str):
        super().__init__(pet, workspace, file, thm)
        self.state: State = self.initial_state
        self.previous_unsuccessful = []

    def parse(self, message: ChatCompletionMessage) -> list[str]:
        """
        Parse the LLM response.
        Handle multi-steps (e.g., `/step intros. [...] /step induction n.`).
        Handle multi-tactics step (e.g., `/step intros. induction n.`)
        """
        if not message.content:
            return []
        else:
            pattern = r"/step\b(.*)"
            steps = [match.group(1) for match in re.finditer(pattern, message.content)]
            return [
                tac for step in steps for tac in split_proof(step, add_delimiter=False)
            ]

    @property
    def proof_finished(self) -> bool:
        # Hack to bypass Petanque proof_finished flag
        try:
            self.pet.run_tac(self.state, "Qed.")
            return True
        except PetanqueError:
            return False

    def exec(self, message: ChatCompletionMessage):
        """
        Parse and execute the LLM message.
        Keep partial progresses if execution fails in the middle of a multi-steps / multi-tactics command.
        """
        self.n_interactions += 1
        tactics = self.parse(message)
        for tac in tactics:
            try:
                self.state = self.pet.run_tac(self.state, tac)
                self.previous_unsuccessful = []
                self.proof.append(tac)
            except PetanqueError as err:
                # Add error message to the prompt ?
                self.previous_unsuccessful.append(tac)
                break

    @property
    def prompt(self) -> Iterable[ChatCompletionMessageParam]:
        """
        Build the tactic prompt from the current goal.
        """
        context = tactic_prompts.system_prompt
        content = tactic_prompts.user_prompt.format(
            n_interactions=self.n_interactions,
            theorem_code=self.thm_code,
            proof_steps=self.proof,
            previous_unsuccessful="\n".join(self.previous_unsuccessful),
            current_goal=pp_goals(self.pet.goals(self.state)),
        )
        return [
            {"role": "system", "content": context},
            {"role": "user", "content": content},
        ]


@dataclass
class Template:
    state: State
    proof: list[Union[str, "Template"]] = field(default_factory=list)

    def __repr__(self) -> str:
        if not self.proof:
            return '"admit."'
        return "[" + ", ".join(step.__repr__() for step in self.proof) + "]"

    @property
    def tactics(self) -> list[str]:
        res = []
        for step in self.proof:
            match step:
                case str():
                    res += [step]
                case Template():
                    if step.proof:
                        res += step.tactics
                    else:
                        res += ["admit."]
        return res


class TemplateEnv(Env):
    """
    Hierarchical proof templating agent
    """

    def __init__(self, pet: Pytanque, workspace: str, file: str, thm: str):
        super().__init__(pet, workspace, file, thm)
        self.template = Template(state=self.initial_state, proof=[])
        self.holes: deque[Template] = deque([self.template])

    def parse(self, message: ChatCompletionMessage) -> str:
        if message.content:
            tag_pattern = r"<proof>(.*?)</proof>"
            if match := re.search(tag_pattern, message.content, re.DOTALL):
                return match.group(1).strip()
            md_pattern = r"```(?:coq)?(.*?)```"
            if match := re.search(md_pattern, message.content, re.DOTALL):
                return match.group(1).strip()
        return "admit."  # cannot parse proof

    def templatize(self, state: State, proof: str) -> Tuple[Template, list[Template]]:
        """
        Return the template of a proof with holes, and a list of pointers to the holes.
        """

        template = Template(state=state, proof=[])
        holes = []

        def fix(
            state: State, tactics: list[str], drop: bool
        ) -> Tuple[Template, list[Template]]:
            if not tactics:
                if not self.pet.goals(state):
                    return template, holes
                else:
                    # This should never happen
                    raise PetanqueError(-1, "Incomplete template")

            tac = tactics[0]
            if tac in ["Abort.", "Admitted."]:
                # Replace by admit and continue to fix the end of the template
                tac = "admit."
            try:
                next_state = self.pet.run_tac(state, tac)
                if tac in ["admit.", "give_up."]:
                    h = Template(state)
                    template.proof.append(h)
                    holes.append(h)
                else:
                    template.proof.append(tac)
                return fix(next_state, tactics[1:], False)

            except PetanqueError as err:
                if drop:  # still invalid, drop tactic.
                    return fix(state, tactics[1:], True)
                if m := re.match(
                    r"Coq: \[Focus\] Wrong bullet (?:\++|\-+|\*+): Expecting (?P<bullet>\++|\-+|\*+)",
                    err.message,
                ):  # refocus on correct bullet
                    return fix(state, [m.group("bullet")] + tactics, False)
                if m := re.match(
                    r"Coq: No such goal. Focus next goal with bullet (?P<bullet>\++|\-+|\*+)",
                    err.message,
                ):  # refocus on correct bullet
                    return fix(state, [m.group("bullet")] + tactics, False)
                if re.match(
                    r"Coq: \[Focus\] Wrong bullet (?:\++|\-+|\*+): Current bullet (?:\++|\-+|\*+) is    not finished.",
                    err.message,
                ):  # close previous subgoal and retry.
                    return fix(state, ["admit."] + tactics, False)
                if re.match(
                    r"Coq: This proof is focused, but cannot be unfocused this way",
                    err.message,
                ):  # close current goal and try to unfocus
                    return fix(state, ["admit."] + tactics, False)
                if re.match(
                    r"Coq: \[Focus\] Wrong bullet (?:\++|\-+|\*+): No more goals.",
                    err.message,
                ):  # Drop bullet
                    return fix(state, tactics[1:], True)
                else:  # replace tac by admit and drop until next valid tactic.
                    return fix(state, ["admit."] + tactics[1:], True)

        tactics = split_proof(proof, add_delimiter=True)
        return fix(state, tactics, False)

    @property
    def proof_finished(self) -> bool:
        """
        Double check the proof, re-running all tactics from the initial state.
        """
        if self.holes:
            return False
        try:  # double check the proof
            s = self.initial_state
            for tac in self.proof:
                s = self.pet.run_tac(s, tac)
            self.pet.run_tac(s, "Qed.")
            return True
        except PetanqueError:
            return False

    def exec(self, message: ChatCompletionMessage):
        """
        Parse and templatize the LLM message to fill the first hole.
        Append the generated holes at the end of the queue.
        """
        self.n_interactions += 1
        h = self.holes.popleft()
        proof = self.parse(message)
        sub_template, sub_holes = self.templatize(h.state, proof)
        if sub_template.tactics == ["{", "admit.", "}"]:  # Remove nested admit.
            h.proof = ["admit."]
            self.holes.extend(sub_holes)
        else:
            h.proof = sub_template.proof
            self.holes.extend(sub_holes)
        self.proof = self.template.tactics

    @property
    def prompt(self) -> Iterable[ChatCompletionMessageParam]:
        """
        Build the template prompt from the first hole
        """
        if self.holes:
            h = self.holes[0]
            content = template_prompts.user_prompt.format(
                context="",
                theorem_name=self.thm,
                theorem_code=self.thm_code,
                goal=pp_goals(self.pet.goals(h.state)),
            )
            return [{"role": "user", "content": content}]
        else:
            raise RuntimeError("No more holes")
