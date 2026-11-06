"""Transition distillation: one function written from every step of every run, failures included."""

TRANSITION_PROMPT = """{contract}

Below are all of the assistant's runs at this job, accepted and rejected:
each request, every tool call with its result, including rejected bookings
and errors, and the user's verdict with its reason. Write the function so
it finds what the accepted runs found and avoids every mistake the
rejected runs made.

{episodes}"""


async def learn(lab):
    shown = "\n\n".join(lab.render(e) for e in lab.episodes)
    return await lab.write(lab.fill(TRANSITION_PROMPT, contract=lab.contract, episodes=shown))
