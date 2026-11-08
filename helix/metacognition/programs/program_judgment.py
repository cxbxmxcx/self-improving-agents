"""Program with judgment: as program synthesis, but what a request is asking for is Jev's call, never a word list."""

DRAFT_PROMPT = """{contract}

Below are runs where the assistant, a language model, handled requests for
this job: each request, every tool call with its result, and the verdict.
Some requests are worded oddly, some should not be acted on at all, and
some runs went wrong. Write a decision program that handles requests the
way the accepted runs did, with no language model.

Split the work this way. Code finds the ids in a request, whatever form
they come in, works out from these runs how ids are really written in the
tools, and calls the procedures and actions. Jev decides what the request
is asking for. That is a judgment, and words can't settle it: a request
can use all of a job's own words while asking whether it was done, telling
the assistant not to do it, asking only for information, or asking to undo
or change it. So never test for words or phrases to tell what a request
asks. Always ask Jev, with a choice over the jobs and a noul for whether
the request asks to carry out that job now, and act only when both are
sure. When a request names several ids, ask Jev which one it means.
Hand off anything else: a wrong action costs far more than a hand-off.

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
Decide what a request asks for with Jev, never with a list of words, and
improve the questions you ask Jev when it was the judgment that failed.
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
