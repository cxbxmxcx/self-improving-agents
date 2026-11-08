"""Reflection: use the function, write a lesson from each rejection, revise to follow every lesson."""

# Past a handful of revisions, lessons start to repeat and the function drifts rather than improves.
MAX_REVISIONS = 5

SUCCESS_PROMPT = """{contract}

Below are runs where the assistant did this job itself and the user
accepted the booking: each request, every tool call with its result, and
the verdict. Write the function so that it would have found what these
runs found.

{episodes}"""

LESSON_PROMPT = """A calendar assistant used the function below for this request:
{request}

It went wrong: {reason}

The calendars it read:
{calendars}

```python
{source}
```

In one or two sentences, state the lesson: what the function gets wrong
about calendars like these. Name the general rule, not this request."""

REVISE_PROMPT = """{contract}

The current function:

```python
{source}
```

Lessons learned from requests where it went wrong:
{lessons}

Rewrite the function so it follows every lesson."""


async def learn(lab):
    successes = [e for e in lab.episodes if e.success]
    first = successes[0] if successes else lab.episodes[0]
    source = await lab.write(lab.fill(SUCCESS_PROMPT, contract=lab.contract, episodes=lab.render(first)))
    lessons = []
    for task_id in lab.stream():
        if len(lessons) >= MAX_REVISIONS:
            break
        reward, reason, calendars = await lab.practice(source, task_id)
        if reward >= 1.0:
            continue
        lesson = await lab.ask(lab.fill(LESSON_PROMPT, request=lab.request(task_id), reason=reason,
                                        calendars=calendars, source=source))
        lessons.append(lesson.strip())
        lab.note(lesson.strip())
        revised = lab.fill(REVISE_PROMPT, contract=lab.contract, source=source,
                           lessons="\n".join(f"- {item}" for item in lessons))
        try:
            source = await lab.write(revised)
        except lab.Refused:
            continue
    return source
