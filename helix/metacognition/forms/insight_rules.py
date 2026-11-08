"""Insight rules: short rules that separate the accepted runs from the rejected ones (ExpeL), edited from practice."""

RULES_PROMPT = """{contract}

Below are the assistant's runs at this job, accepted and rejected: each
request, every tool call with its result, and the verdict with its
reason. Compare them and write the rules that separate the accepted
runs from the rejected ones: short, numbered, general rules the
assistant can apply to new requests of this job, such as which values
to use, which exceptions apply, and how to round. Write the rules only,
with no code and no examples.

{episodes}"""

EDIT_PROMPT = """The assistant applied these rules:

{rules}

and these requests still came out wrong:

{failures}

Edit the rules: add a rule for each mistake, change a rule that misled
the assistant, and drop a rule that never helped. Keep them short and
numbered. Write the rules only."""

HEADER = "Rules for this job, learned from earlier requests:\n\n"
ROUNDS = 2
PRACTICE = 8


async def learn(lab):
    shown = "\n\n".join(lab.render(e) for e in lab.episodes)
    rules = await lab.ask(lab.fill(RULES_PROMPT, contract=lab.contract, episodes=shown))
    best = rules
    best_score = -1.0
    for attempt in range(ROUNDS):
        failures = []
        score = 0.0
        for task_id in lab.stream()[:PRACTICE]:
            reward, reason, observed = await lab.practice(HEADER + rules, task_id)
            score += reward
            if reward < 1:
                failures.append(lab.request(task_id) + "\n  -> " + reason)
        if score > best_score:
            best = rules
            best_score = score
        if not failures:
            break
        rules = await lab.ask(lab.fill(EDIT_PROMPT, rules=rules, failures="\n\n".join(failures)))
    return HEADER + best
