system_prompt = """
Complete a proof using the Coq proof assistant. For each theorem, you will be given the goal to prove in Coq syntax. 

First, analyze and explain the goal. Describe the proof step-by-step, outlining the logical approach needed to reach the conclusion. Finally, write the corresponding Coq proof using your analysis. Do not repeat the context or restate the theorem. 

# Steps

1. **Goal Explanation**: Start by interpreting and explaining the given goal in a concise manner.
2. **Proof Strategy**: Outline your method of approach to solve the goal. Mention useful tactics or transformations that will aid in proving the theorem.
3. **Proof Narration**: Break down each step in the logical process that leads to the proof, explaining your reasoning.
4. **Coq Syntax**: Write the final Coq proof using commands reflecting your step-by-step reasoning.

# Output Format

- Begin with an explanation of the goal.
- Provide a detailed proof strategy.
- Narrate each step of the proof logically.
- Conclude with the Coq code that completes the proof in standard syntax. 

# Examples

**Example 1**

- **Goal**: 
```
n, m, p : nat |- nat, n + (m + p) = m + (n + p)
```
- **Explanation**: You want to show that addition is associative for natural numbers. This means the grouping of numbers in addition does not affect the result.
- **Proof Strategy**: Use associativity and commutativity properties of addition.
- **Proof Narration**: 
  1. Exploit associativity by applying `Nat.add_assoc` to shift brackets.
  2. Use the commutative property to rearrange terms.
  3. Close the proof with `reflexivity` to establish equality.
- **Coq Proof**:
  ```coq
  rewrite Nat.add_assoc.
  rewrite Nat.add_assoc.
  assert (n + m = m + n) as H by apply Nat.add_comm.
  rewrite H.
  reflexivity.
  ```

**Example 2**

- **Goal**: 
```
|- forall n:nat, n + 0 = n
```
- **Explanation**: The goal demonstrates that adding zero to any natural number leaves the number unchanged.
- **Proof Strategy**: Use induction on `n`.
- **Proof Narration**:
  1. Start with base case where `n = 0`.
  2. Use induction hypothesis to prove the step case by simplifying the equation.
- **Coq Proof**:
  ```coq
  intros n.
  induction n as [| n' IHn'].
  - reflexivity.
  - simpl.
    rewrite -> IHn'.
    reflexivity.
  ```

**Example 3**

- **Goal**: 
```
f nat -> nat 
I: forall n : nat, n = f (f n) 
n1, n2 : nat 
H: f n1 = f n2 
|- n1 = n2
```
- **Explanation**: You need to show that given `f n1 = f n2` and the invariant `I` for `f`, their respective inputs are equal.
- **Proof Strategy**: Substitute using `I` and `H` to reach the result.
- **Proof Narration**:
  1. Replace `n1` with its double application using `I`.
  2. Apply `H` and revert the substitution using `I`.
  3. Conclude with `reflexivity`.
- **Coq Proof**:
  ```coq
  rewrite (I n1).
  rewrite H.
  rewrite <- I.
  reflexivity.
  ```
  """

user_prompt = """
{decorated_context}

You are in the middle of the proof of {theorem_name}:

{theorem_code}

Ready? Here is the current goal.
<goal>
{goal}
</goal>

Take a deep breath and walk me through the proof (do not repeat the context or restate the theorem, just the proof steps).
"""

display_user_prompt = """## Template prompt
theorem name: {theorem_name}
theorem code: {theorem_code}
goal: {goal}
"""

decorated_context = """
Here are some useful definitions and lemmas:

{context}
"""

comparison_system_prompt = """
For each attempt, you will receive the current proof and the number of interactions. Use this information to decide on the likeliness of the proof succeeding, factoring in other proof strategies if necessary.

# Steps

1. **Analyze Each Attempt:** 
   - Examine the given theorem and each proof attempt.
   - Consider the structure and progress of each proof.
   - Evaluate the number of interactions to assess complexity and tractability.

2. **Assess Likelihood of Success:**
   - Determine which proofs are closer to completion or seem more strategically sound.
   - Rank the attempts based on these assessments.

3. **Consider Alternative Strategies:**
   - Identify attempts that seem overly complex and could benefit from a different proof strategy.

# Output Format

First, provide a thorough rationale describing the analysis and assessment process. Then, offer the final ranking of the proof attempts as a list, ordered from most likely to succeed to least likely.

Example format:

- **Reasoning:** A detailed rationale of the evaluation for each proof attempt, discussing the strengths and weaknesses, and considering whether the number of interactions suggests the proof is manageable or overly complex.
- **Response:** A list of indices representing the ranking of attempts, e.g., [2, 0, 1, 3].

# Notes

- Provide a clear and logical reasoning process before concluding with a ranking.
- Consider the possibility of both technical soundness and the tractability of each proof attempt.
- Assumptions about the quality of the proof should be based on your knowledge of mathematical logic and Coq.
"""

prompt_for_comparison = """
### Proof

Here is the current template proof until now:

{template_proof}

You have interacted {n_interactions} times with the engine.

"""

display_prompt_for_comparison = """## Comparison prompt
template proof: {template_proof}
n interactions: {n_interactions}
"""
