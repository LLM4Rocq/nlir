import re
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import deque
from typing import Iterable, Union, Tuple
from pytanque import Pytanque, State, Goal, PetanqueError

# from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessage
from .agent import UserMessage, SystemMessage, Message, Response
from .prompts import tactic_prompts, template_prompts
import copy
import weave


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


def remove_comments(code: str) -> str:
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
        for a in re.split(r"(\{|\})", raw)  # split braces
        for b in re.split(r"(?<=\.)\s+(\++|\*+|\-+)", a)  # split proof in bullets
        for s in re.split(r"(?<=\.)\s+", b)  # split bullets in tactics
        if (t := s.strip())  # remove empty steps
    ]
    if add_delimiter:
        return ["{", *tactics, "}"]
    return tactics


def get_context(doc: str, thm: str) -> str:
    """
    Remove all proof to get context
    """
    pattern = r"Proof\.(.*?)(Qed|Admitted|Abort)\."
    cleaned_text = re.sub(pattern, "", doc, flags=re.DOTALL)
    # Replace multiple newlines with a single newline
    cleaned_text = re.sub(r"\n+", "\n", cleaned_text)
    lines = cleaned_text.split("\n")
    for i, l in enumerate(lines):
        if thm in l:
            return "\n".join(lines[:i])
    return cleaned_text


# Dictionnary containing libraries providing automatic solver tactics,
# how to import them,
# and the tactics they introduce.
auto_libraries = {
    "Init.Ltac": {
        "import": "",
        "tactics": ["auto", "eauto"]
    },
    "Lra": {
        "import": "Require Import Lra.",
        "tactics": ["lra", "nra"]
    },
    "Lia": {
        "import": "Require Import Lia.",
        "tactics": ["lia", "nia"]
    },
    "Lqa": {
        "import": "Require Import Lqa.",
        "tactics": ["lra", "nra"]
    },
    "Psatz": {
        "import": "Require Import Psatz.",
        "tactics": ["psatz"]
    },
    "Hammer.Tactics.Tactics": {
        "import": "From Hammer Require Import Tactics.",
        "tactics": ["sauto", "hauto"]
    }
}

# A dictionnary mapping automatic solver tactics to their possible imports.
auto_tactics_index = {}
for lib in auto_libraries.values():
    for tactic in lib["tactics"]:
        info = {"lib" : lib, "import": lib["import"]}
        if tactic in auto_tactics_index:
            auto_tactics_index[tactic].append(info)
        else:
            auto_tactics_index[tactic] = [info]

# Dictionnary containing libraries providing tactics,
# how to import them,
# and the tactic they introduce.
other_libraries = {}

# A dictionnary mapping all tactics in given libraries to their possible imports.
tactics_index = auto_tactics_index.copy()
for lib in other_libraries.values():
    for tactic in lib["tactics"]:
        info = {"lib" : lib, "import": lib["import"]}
        if tactic in tactics_index:
            tactics_index[tactic].append(info)
        else:
            tactics_index[tactic] = [info]

def fix_tactic_import(tactic: str) -> list[tuple[str, str]]:
    """
    Add the possible imports before a tactic.
    """
    if tactic in tactics_index:
        if len(tactics_index[tactic]) > 1:
            return tactics_index[tactic].map(lambda info:
                    (info["import"], info["lib"] + "." + tactic)
                )
        else:
            info = tactics_index[tactic][0]
            return [(info["import"], tactic)]
    else:
        return []


@dataclass
class Status:
    success: bool
    state: State
    proof: list[str]

class Env(ABC):
    """
    Base class for a Petanque environment.
    `exec` and `output` need to be defined for `search`
    """

    def __init__(self, pet, workspace, file, thm, context, verbose=False):
        self.pet = pet
        self.workspace = workspace
        self.file = file
        self.path = os.path.join(workspace, file)
        self.thm = thm
        self.proof: list[str] = []
        self.initial_state: State = self.pet.start(self.path, thm)
        self.thm_code = pp_goals(self.pet.goals(self.initial_state))
        self.n_interactions = 0
        self.verbose = verbose
        if context:
            with open(self.path, "r") as read_file:
                self.context = get_context(read_file.read(), thm)
        else:
            self.context = None

    @property
    @abstractmethod
    def proof_finished(self) -> bool:
        pass

    @abstractmethod
    def exec(self, response: Response):
        pass

    @property
    @abstractmethod
    def prompt(self) -> list[Message]:
        pass

    def check_proof(self) -> bool:
        """
        Double check the proof, re-running all tactics from the initial state.
        """
        try:  # double check the proof
            s = self.initial_state
            for tac in self.proof:
                s = self.pet.run_tac(s, tac)
            self.pet.run_tac(s, "Qed.")
            return True
        except PetanqueError:
            return False

    def deepcopy(self):
        new = self.__class__(
            self.pet, self.workspace, self.file, self.thm, self.context
        )
        new.proof = copy.deepcopy(self.proof)
        new.n_interactions = copy.deepcopy(self.n_interactions)
        return new

    @property
    @abstractmethod
    def info_for_comparison(self) -> str:
        pass

    def try_automatic_solving(self, state) -> Status:
        """
        Try resulving the given state with the Rocq automatic solver tactics.
        """
        for tactic in auto_tactics_index.keys():
            for imp, tactic in fix_tactic_import(tactic):
                try:
                    new_state = self.pet.run_tac(state, imp, timeout=10)
                    new_state = self.pet.run_tac(new_state, tactic, timeout=10)
                    if self.verbose:
                        print("success:", tactic)
                    return Status(success=True, state=new_state, proof=[imp, tactic])
                except PetanqueError as err:
                    if self.verbose:
                        print(tactic, "->", err.message)
        return Status(state, success=False, proof=[])


    def automatic_solving(self) -> tuple[bool, str]:
        """
        Try resolving the proof with the Rocq automatic solver tactics.
        """
        status = self.try_automatic_solving(self.state)
        return status.success, " ".join(status.proof)


class TacticEnv(Env):
    """
    Tactic-by-tactic proof construction.
    """

    def __init__(
        self, pet: Pytanque, workspace: str, file: str, thm: str, context: bool
    ):
        super().__init__(pet, workspace, file, thm, context)
        self.state: State = self.initial_state
        self.previous_unsuccessful = []

    def parse(self, response: Response) -> list[str]:
        """
        Parse the LLM response.
        Handle multi-steps (e.g., `/step intros. [...] /step induction n.`).
        Handle multi-tactics step (e.g., `/step intros. induction n.`)
        """
        if not response.content:
            return []
        else:
            pattern = r"/step\b(.*)"
            steps = [match.group(1) for match in re.finditer(pattern, response.content)]
            if self.verbose:
                print(
                    "parsed tactics:",
                    [
                        tac
                        for step in steps
                        for tac in split_proof(step, add_delimiter=False)
                    ],
                )
            return [
                tac for step in steps for tac in split_proof(step, add_delimiter=False)
            ]

    def deepcopy(self):
        new = super().deepcopy()
        new.state = copy.deepcopy(self.state)
        new.previous_unsuccessful = copy.deepcopy(self.previous_unsuccessful)
        return new

    @property
    def proof_finished(self) -> bool:
        # Hack to bypass Petanque proof_finished flag
        try:
            self.pet.run_tac(self.state, "Qed.")
            return True
        except PetanqueError:
            return False

    def exec_tactic(self, state, tac, imp=None) -> bool:
        """
        Execute one tactic with a potential import tactic before.
        """
        try:
            if imp:
                state = self.pet.run_tac(self.state, imp, timeout=10)
            else:
                state = self.state
            self.state = self.pet.run_tac(state, tac, timeout=10)
            self.previous_unsuccessful = []
            if imp:
                self.proof.append(imp)
            self.proof.append(tac)
            if self.verbose:
                print("success")
            return True
        except PetanqueError as err:
            if m := re.match(
                r"Coq: The reference (?P<tactic>\S*) was not found in the current environment",
                err.message
            ):
                fixed_tactics = fix_tactic_import(tac)
                if fixed_tactics:
                    for imp, tac in fixed_tactics:
                        if self.exec_tactic(tac, imp):
                            return True
                    return False
            if self.verbose:
                print("error", err.message)
            self.previous_unsuccessful.append(str(tac) + str(err.message))
            return False

    def exec(self, response: Response):
        """
        Parse and execute the LLM message.
        Keep partial progresses if execution fails in the middle of a multi-steps / multi-tactics command.
        """
        self.n_interactions += 1
        tactics = self.parse(response)
        for tac in tactics:
            if self.verbose:
                print("tactic:", tac)
            if not self.exec_tactic(tac):
                break

    @property
    def prompt(self) -> list[Message]:
        """
        Build the tactic prompt from the current goal.
        """
        syst_prompt = tactic_prompts.system_prompt
        if self.context:
            decorated_context = tactic_prompts.decorated_context.format(
                context=self.context
            )
        else:
            decorated_context = ""
        content = tactic_prompts.user_prompt.format(
            n_interactions=self.n_interactions,
            decorated_context=decorated_context,
            theorem_code=self.thm_code,
            proof_steps=self.proof,
            previous_unsuccessful="\n".join(self.previous_unsuccessful),
            current_goal=pp_goals(self.pet.goals(self.state)),
        )
        if self.verbose:
            print(
                tactic_prompts.display_user_prompt.format(
                    n_interactions=self.n_interactions,
                    theorem_code=self.thm_code,
                    proof_steps=self.proof,
                    previous_unsuccessful="\n".join(self.previous_unsuccessful),
                    current_goal=pp_goals(self.pet.goals(self.state)),
                )
            )
        return [
            SystemMessage(role="system", content=syst_prompt),
            UserMessage(role="user", content=content),
        ]

    @property
    def info_for_comparison(self) -> str:
        """
        Build the string containing the informations to be included in the comparison prompt.
        """
        if self.verbose:
            print(
                tactic_prompts.display_prompt_for_comparison.format(
                    current_goal=pp_goals(self.pet.goals(self.state)),
                    proof_steps=self.proof,
                    n_interactions=self.n_interactions,
                    previous_unsuccessful="\n".join(self.previous_unsuccessful),
                )
            )
        return tactic_prompts.prompt_for_comparison.format(
            current_goal=pp_goals(self.pet.goals(self.state)),
            proof_steps=self.proof,
            n_interactions=self.n_interactions,
            previous_unsuccessful="\n".join(self.previous_unsuccessful),
        )


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

    def __init__(
        self, pet: Pytanque, workspace: str, file: str, thm: str, context: bool
    ):
        super().__init__(pet, workspace, file, thm, context)
        self.template = Template(state=self.initial_state, proof=[])
        self.holes: deque[Template] = deque([self.template])

    def deepcopy(self):
        new = super().deepcopy()
        memo = {}
        new.template = copy.deepcopy(self.template, memo)
        new.holes = copy.deepcopy(self.holes, memo)
        return new

    def parse(self, response: Response) -> str:
        if response.content:
            tag_pattern = r"<proof>(.*?)</proof>"
            if match := re.search(tag_pattern, response.content, re.DOTALL):
                return match.group(1).strip()
            md_pattern = r"```(?:coq)?(.*?)```"
            if match := re.search(md_pattern, response.content, re.DOTALL):
                return match.group(1).strip()
        return "admit."  # cannot parse proof

    def templatize(self, state: State, proof: str) -> Tuple[Template, list[Template]]:
        """
        Return the template of a proof with holes, and a list of pointers to the holes.
        """

        template = Template(state=state, proof=[])
        holes = []

        def fix(
            state: State, tactics: list[str], opened_par: int, drop: bool
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
                next_state = self.pet.run_tac(state, tac, timeout=10)
                if tac in ["admit.", "give_up."]:
                    h = Template(state)
                    template.proof.append(h)
                    holes.append(h)
                elif tac == "{":
                    opened_par += 1
                elif tac == "}":
                    if opened_par == 0:
                        raise PetanqueError(-32003, "Coq: The proof is not focused")
                    opened_par -= 1
                else:
                    template.proof.append(tac)
                return fix(next_state, tactics[1:], opened_par, False)

            except PetanqueError as err:
                # print("xxxx", err.message)
                if drop:  # still invalid, drop tactic.
                    return fix(state, tactics[1:], opened_par, True)
                if m := re.match(
                    r"Coq: \[Focus\] Wrong bullet (?:\++|\-+|\*+): Expecting (?P<bullet>\++|\-+|\*+)",
                    err.message,
                ):  # refocus on correct bullet
                    return fix(state, [m.group("bullet")] + tactics, opened_par, False)
                if m := re.match(
                    r"Coq: No such goal. Focus next goal with bullet (?P<bullet>\++|\-+|\*+)",
                    err.message,
                ):  # refocus on correct bullet
                    return fix(state, [m.group("bullet")] + tactics, opened_par, False)
                if re.match(
                    r"Coq: \[Focus\] Wrong bullet (?:\++|\-+|\*+): Current bullet (?:\++|\-+|\*+) is not finished.",
                    err.message,
                ):  # close previous subgoal and retry.
                    return fix(state, ["admit."] + tactics, opened_par, False)
                if re.match(
                    r"Coq: This proof is focused, but cannot be unfocused this way",
                    err.message,
                ):  # close current goal and try to unfocus
                    return fix(state, ["admit."] + tactics, opened_par, False)
                if re.match(
                    r"Coq: \[Focus\] Wrong bullet (?:\++|\-+|\*+): No more goals.",
                    err.message,
                ):  # Drop bullet
                    return fix(state, tactics[1:], opened_par, True)
                if m := re.match(
                    r"Coq: The reference (?P<tactic>\S*) was not found in the current environment",
                    err.message
                ):  # try importing the correct library for the tactic
                    for imp, tac in fix_tactic_import(tac):
                        try:
                            new_state = self.pet.run_tac(state, imp, timeout=10)
                            new_state = self.pet.run_tac(new_state, tac, timeout=10)
                            template.proof.append(imp)
                            template.proof.append(tac)
                            fix(new_state, tactics[1:], opened_par, drop)
                        except PetanqueError as err:
                            pass
                    return fix(state, ["admit."] + tactics[1:], opened_par, True)
                else:  # replace tac by admit and drop until next valid tactic.
                    return fix(state, ["admit."] + tactics[1:], opened_par, True)

        tactics = split_proof(proof, add_delimiter=True)
        try:
            return fix(state, tactics, 0, False)
        except RecursionError:
            # Proof requires too many fixes (maybe it generates too many subgoals?)
            # Return a template with a single hole.
            h = Template(state)
            template.proof = [h]
            holes = [h]
            return template, holes

    @property
    def proof_finished(self) -> bool:
        if self.holes:
            return False
        return self.check_proof()

    def exec(self, response: Response):
        """
        Parse and templatize the LLM message to fill the first hole.
        Append the generated holes at the end of the queue.
        """
        self.n_interactions += 1
        h = self.holes.popleft()
        proof = self.parse(response)
        sub_template, sub_holes = self.templatize(h.state, proof)
        if sub_template.tactics == ["{", "admit.", "}"]:  # Remove nested admit.
            h.proof = []
            self.holes.append(h)
        else:
            h.proof = sub_template.proof
            self.holes.extend(sub_holes)
        self.proof = self.template.tactics

    @property
    def prompt(self) -> list[Message]:
        """
        Build the template prompt from the first hole
        """
        if self.holes:
            h = self.holes[0]
            if self.context:
                decorated_context = template_prompts.decorated_context.format(
                    context=self.context
                )
            else:
                decorated_context = ""
            content = template_prompts.user_prompt.format(
                decorated_context=decorated_context,
                theorem_name=self.thm,
                theorem_code=self.thm_code,
                goal=pp_goals(self.pet.goals(h.state)),
            )
            if self.verbose:
                print(
                    template_prompts.display_user_prompt.format(
                        theorem_name=self.thm,
                        theorem_code=self.thm_code,
                        goal=pp_goals(self.pet.goals(h.state)),
                    )
                )
            return [
                SystemMessage(role="system", content=template_prompts.system_prompt),
                UserMessage(role="user", content=content),
            ]
        else:
            raise RuntimeError("No more holes")

    @property
    def info_for_comparison(self) -> str:
        """
        Build the string containing the informations to be included in the comparison prompt.
        """
        if self.verbose:
            print(
                template_prompts.display_prompt_for_comparison.format(
                    template_proof=self.template.tactics,
                    n_interactions=self.n_interactions,
                )
            )
        return template_prompts.prompt_for_comparison.format(
            template_proof=self.template.tactics,
            n_interactions=self.n_interactions,
        )
