"""Skill synthesis: write from the accepted runs, execute on the observed requests, repair what failed."""

ROUNDS = 3

SYNTHESIS_PROMPT = """{contract}

Below are runs where the assistant did this job itself and the user
accepted the booking: each request, every tool call with its result, and
the verdict. Write the function so that it would have found what these
runs found. It will then be executed on these same requests, and you will
see where it goes wrong.

{episodes}"""

REPAIR_PROMPT = """{contract}

The current function:

```python
{source}
```

Executed on the requests the assistant has seen, it went wrong on these:
{failures}

Fix the function so these are accepted, without breaking the others."""


async def learn(lab):
    shown = "\n\n".join(lab.render(e) for e in lab.episodes if e.success)
    source = await lab.write(lab.fill(SYNTHESIS_PROMPT, contract=lab.contract, episodes=shown))
    for round_number in range(ROUNDS):
        failures = []
        for task_id in lab.stream():
            reward, reason, calendars = await lab.practice(source, task_id)
            if reward < 1.0:
                failures.append(f"- {lab.request(task_id)} {reason}")
        lab.note(f"{len(failures)} rejected")
        if not failures:
            break
        prompt = lab.fill(REPAIR_PROMPT, contract=lab.contract, source=source, failures="\n".join(failures))
        try:
            source = await lab.write(prompt)
        except lab.Refused:
            break
    return source
