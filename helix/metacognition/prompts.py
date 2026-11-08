"""Prompt packs: the words a learning mechanism writes with.

The book's writer prompts speak of a calendar assistant, bookings, and
calendars. They are kept verbatim as CALENDAR, taken from helix itself, so the
book's recordings replay. GENERIC says the same things without a domain and
is the default for any other agent.

Placeholders: contract takes {signature} {description} {api} {returns}
{builtins}; the rest take {contract} plus {episodes}, {source}, {lessons},
{failures}, or, for lesson, {request} {reason} {source} and the run's
observations as {observations} (CALENDAR names them {calendars}).
"""

from __future__ import annotations

from dataclasses import dataclass

from helix.memory import learning_mechanisms as book


@dataclass(frozen=True)
class PromptPack:
    name: str
    contract: str
    no_experience: str
    success: str
    transition: str
    lesson: str
    revise: str
    synthesis: str
    repair: str


CALENDAR = PromptPack(
    name="calendar",
    contract=book.CONTRACT_PROMPT,
    no_experience=book.NO_EXPERIENCE_PROMPT,
    success=book.SUCCESS_PROMPT,
    transition=book.TRANSITION_PROMPT,
    lesson=book.LESSON_PROMPT,
    revise=book.REVISE_PROMPT,
    synthesis=book.SYNTHESIS_PROMPT,
    repair=book.REPAIR_PROMPT,
)

GENERIC = PromptPack(
    name="generic",
    contract="""You are writing a Python function that an assistant will call
instead of working the answer out itself. Its contract:

```python
{signature}
    \"\"\"{description}\"\"\"
```

{api} {returns}

Rules: no imports, no names that begin with an underscore, and plain Python
only: functions, loops, comprehensions, methods of str, int, list, and dict,
and only these builtins: {builtins}. Reply with one ```python block that
holds the whole function.""",
    no_experience="{contract}\n\nWrite the function.",
    success="""{contract}

Below are runs where the assistant did this job itself and the outcome was
accepted: each request, every tool call with its result, and the verdict.
Write the function so that it would have found what these runs found.

{episodes}""",
    transition="""{contract}

Below are all of the assistant's runs at this job, accepted and rejected:
each request, every tool call with its result, including rejected actions
and errors, and the verdict with its reason. Write the function so it finds
what the accepted runs found and avoids every mistake the rejected runs made.

{episodes}""",
    lesson="""An assistant used the function below for this request:
{request}

It went wrong: {reason}

What it observed:
{observations}

```python
{source}
```

In one or two sentences, state the lesson: what the function gets wrong
about inputs like these. Name the general rule, not this request.""",
    revise=book.REVISE_PROMPT,
    synthesis="""{contract}

Below are runs where the assistant did this job itself and the outcome was
accepted: each request, every tool call with its result, and the verdict.
Write the function so that it would have found what these runs found. It
will then be executed on these same requests, and you will see where it
goes wrong.

{episodes}""",
    repair=book.REPAIR_PROMPT,
)

PACKS = {pack.name: pack for pack in (CALENDAR, GENERIC)}
