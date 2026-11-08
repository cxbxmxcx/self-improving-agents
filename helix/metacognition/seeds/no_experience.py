"""No experience: the control, written from the contract alone.

A mechanism that learns nothing from the agent's runs still gets a function
from the writer. Any design must beat this to show it learned from experience.
"""

NO_EXPERIENCE_PROMPT = "{contract}\n\nWrite the function."


async def learn(lab):
    return await lab.write(lab.fill(NO_EXPERIENCE_PROMPT, contract=lab.contract))
