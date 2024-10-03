from .petanque import TacticEnv


def naive_search(agent, env, max_steps: int):
    response = agent.response(env.prompt)  # Initial step
    for step in range(max_steps):
        env.exec(response)
        print(env.proof)
        if env.proof_finished:
            agent.log({"role": "user", "Final Proof": " ".join(env.proof)})
            return {"Proof Finished": env.proof, "steps": step}
        else:
            response = agent.response(env.prompt)

    if not env.proof_finished:
        agent.log({"Partial Proof": " ".join(env.proof)})
        return {"Partial Proof": env.proof}
