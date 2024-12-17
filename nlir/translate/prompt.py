from typing import Iterable
from openai.types.chat import ChatCompletionMessageParam

class Prompt():
    """
    Basic class for prompts.
    """

    def __init__(self, intro, question):
        self.intro = intro
        self.question = question

    def make(self, theorem) -> Iterable[ChatCompletionMessageParam]:
        return [
            {"role": "system", "content": self.intro},
            {"role": "user", "content": self.question(theorem)}
        ]

intro1 = """
    # Coq translating task

    ## Context

    Hello, you are a researcher specialized in formal verification and proof assistants.
    You want to have a specific dataset of theorems in Coq, but for the moment, this dataset only exists in Lean and Isabelle.
    For each theorem, you have a Lean and an Isabelle version along with a natural language description.
    Your goal is to translate theorems one by one to Coq.

    Here are the relevant informations:

    ### Inputs

    You will be provided with:
    - This information prompt;
    - The natural language description of the theorem to be translated
    - The Lean code for the theorem to be translated;
    - The Isabelle code for the theorem to be translated;

    ### Instructions

    Start by looking at the natural language description of the theorem to be translated to understand what it is about.
    Then, based on the Lean and Isabelle version of the theorem, try to write the theorem in Coq.
    Try to be as concise as possible, go straight to the point.
    Ideally, the Coq code should look like this:
    Theorem "theorem name".
    "body of theorem"
    Proof.
    Admitted.

    As you can see, you must not translate the proof, only the body of the theorem.
    The translation of proofs will be tackled later, don't bother with it now.

    Because the Lean and Isabelle version use existing libraries to express theorems,
    you can do the same in Coq and use any library you judge useful for the translation.
"""

def simple_question(theorem):
    return f"""
        ## Theorem information

        Now, you are going to translate the theorem named {theorem[0]}.

        Here is a description of what the theorem is about:
        {theorem[1]['informal']}

        Here is the code of the theorem in Lean:
        {theorem[1]['lean']}

        And here is the code of the theorem in Isabelle:
        {theorem[1]['isabelle']}

        To help you, describe each step of the Lean and Isabelle versions using the natural language description,
        so you can use those steps when writing the theorem in Coq.

        Take a deep breath and walk me through the process step-by-step.
    """

prompt_list = {
    "simple_question1": Prompt(intro1, simple_question)
}
