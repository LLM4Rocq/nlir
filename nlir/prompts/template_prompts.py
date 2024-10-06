user_prompt = """
Your task is to complete a proof using the Coq proof assistant.
For each theorem, I will give you the goal to prove in Coq syntax.

Here are a few examples:

<example>
<goal>
n, m, p : nat
|- nat, n + (m + p) = m + (n + p)
</goal>

<proof>
rewrite Nat.add_assoc. rewrite Nat.add_assoc.
assert (n + m = m + n) as H by apply Nat.add_comm.
rewrite H. reflexivity.
</proof>
</example>


<example>
<context>
<goal>
|- forall n:nat, n + 0 = n
</goal>

<proof>
intros n. induction n as [| n' IHn'].
- reflexivity.
- simpl. rewrite -> IHn'. reflexivity.
</proof>
</example>

<example>
<goal>
f nat -> nat 
I forall n : nat, n = f (f n) 
n1n2 nat 
H f n1 = f n2 
|- n1 = n2
</goal>

<proof>
rewrite (I n1). rewrite H.
rewrite <- I. reflexivity.
</proof>
</example>

Think before you write the proof in <thinking> tags. First explain the goal. Then describe the proof step by step. Finally write the corresponding Coq proof in <proof> tags using your analysis.
Do not repeat the context and do no restate the theorem.

{context}

You are in the middle of the proof of {theorem_name}:

{theorem_code}


Ready? Here is the current goal.
<goal>
{goal}
</goal>

Take a deep breath and walk me through the proof.
"""


context = """
Here are some useful definitions and lemmas:

{context}
"""
