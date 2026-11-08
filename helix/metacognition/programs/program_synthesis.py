"""Program synthesis: a decision program written from the agent's judged runs, then repaired from its own practice."""

DRAFT_PROMPT = """{contract}

Below are runs where the assistant, a language model, handled requests for
this job: each request, every tool call with its result, and the verdict.
Some requests are worded oddly, some should not be acted on at all, and
some runs went wrong. Write a decision program that handles requests the
way the accepted runs did, with no language model. Work out from these
runs what the program must look for in a request and how ids are really
written in the tools. Use code for everything code can do. Ask Jev only
for judgments code can't make, such as what a request is really asking
for, or which of several candidates it means. Hand off anything you
cannot do with confidence: a wrong action costs far more than a hand-off.

{episodes}"""

REPAIR_PROMPT = """{contract}

This decision program was tried on requests for this job:

```python
{program}
```

These requests came out wrong or were handed off when they could have
been served:

{failures}

Rewrite the program so these come out right, keeping what already works.
Reply with one ```python block that holds the whole program."""

ROUNDS = 3
SHOWN_FAILURES = 14


def code_of(reply):
    parts = reply.split("```")
    if len(parts) < 3:
        return reply.strip() + "\n"
    block = parts[1]
    if block.startswith("python"):
        block = block[len("python"):]
    return block.strip() + "\n"


async def learn(lab):
    shown = "\n\n".join(lab.render(e) for e in lab.episodes)
    program = code_of(await lab.ask(lab.fill(DRAFT_PROMPT, contract=lab.contract, episodes=shown)))
    best = program
    best_score = -1.0
    for attempt in range(ROUNDS):
        failures = []
        score = 0.0
        for task_id in lab.stream():
            reward, reason, observed = await lab.practice(program, task_id)
            score += reward
            if reward < 1:
                failures.append("Request: " + lab.request(task_id) + "\nOutcome: " + reason)
        lab.note("attempt " + str(attempt) + ": score " + str(score))
        if score > best_score:
            best = program
            best_score = score
        if not failures or attempt == ROUNDS - 1:
            break
        program = code_of(await lab.ask(lab.fill(REPAIR_PROMPT, contract=lab.contract, program=program,
                                                 failures="\n\n".join(failures[:SHOWN_FAILURES]))))
    return best
