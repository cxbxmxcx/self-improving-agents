"""Success distillation: one function written from every accepted run."""

SUCCESS_PROMPT = """{contract}

Below are runs where the assistant did this job itself and the user
accepted the booking: each request, every tool call with its result, and
the verdict. Write the function so that it would have found what these
runs found.

{episodes}"""


async def learn(lab):
    shown = "\n\n".join(lab.render(e) for e in lab.episodes if e.success)
    return await lab.write(lab.fill(SUCCESS_PROMPT, contract=lab.contract, episodes=shown))
