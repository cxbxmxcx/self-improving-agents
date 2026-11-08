"""Value learning: chapter 5's reinforcement learning, applied to remembered functions.

One candidate is written from each request the agent solved, from that run
alone. Chapter 5's fixed rule then values each candidate as it is used on
the development requests, and the highest-valued candidate is the memory.
"""

# Chapter 5's learning rule: q_new = q + ALPHA * (credit - q), from a neutral start.
ALPHA = 0.5
INITIAL_VALUE = 0.5
EPOCHS = 2

SUCCESS_PROMPT = """{contract}

Below are runs where the assistant did this job itself and the user
accepted the booking: each request, every tool call with its result, and
the verdict. Write the function so that it would have found what these
runs found.

{episodes}"""


async def learn(lab):
    candidates = {}
    for episode in lab.episodes:
        if episode.success and episode.task_id not in candidates:
            prompt = lab.fill(SUCCESS_PROMPT, contract=lab.contract, episodes=lab.render(episode))
            try:
                candidates[episode.task_id] = await lab.write(prompt)
            except lab.Refused:
                continue
    if not candidates:
        raise ValueError("value learning needs at least one successful run")
    values = {key: INITIAL_VALUE for key in candidates}
    for epoch in range(EPOCHS):
        for task_id in lab.stream():
            # Recall the highest-valued candidate; ties go to the earliest written.
            chosen = max(values, key=lambda k: values[k])
            reward, reason, calendars = await lab.practice(candidates[chosen], task_id)
            values[chosen] += ALPHA * (reward - values[chosen])
            lab.note(f"{task_id}: used {chosen}, reward {reward:g}")
    best = max(values, key=lambda k: values[k])
    return {"source": candidates[best], "values": {k: round(v, 4) for k, v in values.items()}}
