"""Text skill: written instructions the agent follows, drafted from every run and repaired from its own practice."""

SKILL_PROMPT = """{contract}

The assistant will be given the instructions you write whenever it does
this job. It still calls the tools and works out the answer itself.
Below are its runs at this job, accepted and rejected: each request,
every tool call with its result, and the verdict with its reason. Write
step-by-step instructions that would have made every rejected run come
out right and kept every accepted one right: which tools to call, what
to look for in each result, and how to work out the answer. Write the
instructions only, in plain text, with no code.

{episodes}"""

REPAIR_PROMPT = """{contract}

The assistant followed these instructions:

{skill}

and these requests still came out wrong:

{failures}

Rewrite the instructions so those requests come out right, keeping
what already works. Write the instructions only, in plain text, with
no code."""

ROUNDS = 2
PRACTICE = 8


async def learn(lab):
    shown = "\n\n".join(lab.render(e) for e in lab.episodes)
    skill = await lab.ask(lab.fill(SKILL_PROMPT, contract=lab.contract, episodes=shown))
    best = skill
    best_score = -1.0
    for attempt in range(ROUNDS):
        failures = []
        score = 0.0
        for task_id in lab.stream()[:PRACTICE]:
            reward, reason, observed = await lab.practice(skill, task_id)
            score += reward
            if reward < 1:
                failures.append(lab.request(task_id) + "\n  -> " + reason)
        if score > best_score:
            best = skill
            best_score = score
        if not failures:
            break
        skill = await lab.ask(lab.fill(REPAIR_PROMPT, contract=lab.contract, skill=skill,
                                       failures="\n\n".join(failures)))
    return best
