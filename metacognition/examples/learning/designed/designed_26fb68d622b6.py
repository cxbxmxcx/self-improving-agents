"""Transition distillation with API-compatibility guard.

Learn from every judged run, but write a function that uses only supported
surface area and degrades safely when an expected method is absent.
"""

PROMPT = """{contract}

Below are all of the assistant's runs at this job, accepted and rejected:
each request, every tool call with its result, including rejected actions
and errors, and the verdict with its reason.

Write the function by distilling the successful behavior, but be conservative
about external APIs:

- use only objects, attributes, and methods that are explicitly present in the
  contract or demonstrably used in the runs;
- if a needed capability might not exist, probe it with getattr/hasattr/callable
  and fall back safely instead of assuming it;
- do not repeat AttributeError-prone calls from failed runs;
- if the runs reveal alternate method names for the same capability, prefer a
  small compatibility shim over a single hard-coded name;
- preserve the successful logic, but make it robust to missing optional methods.

{episodes}"""


async def learn(lab):
    shown = "\n\n".join(lab.render(e) for e in lab.episodes)
    return await lab.write(lab.fill(PROMPT, contract=lab.contract, episodes=shown))