"""Everything through the language model: memories of how the work went, read by the model that serves each request.

Nothing here serves a request without the model, and nothing here asks Jev. It is where a search starts when the
question is whether a search can find the code that moves work from the model to a decision program.
"""

KINDS_PROMPT = """{contract}

Below are runs where the assistant, a language model, handled requests for
this job: each request, every tool call with its result, and the verdict.
Some requests are worded oddly, some should not be acted on at all, and
some runs went wrong.

List the kinds of request these runs show, and for each write what the
assistant should know the next time it meets one: how ids are really
written in the tools, which tools to call and in what order, and the
mistakes to avoid. Use as few kinds as cover the runs.

Reply with only a JSON list. Each item has the keys "name" (short, in
snake_case), "description" (which requests belong to this kind, and which
similar ones don't), "guidance" (what the assistant should know), and
"acts" (true if a request of this kind should end in an action, false if
it should only be answered or passed on).

{episodes}"""

ROUTER = """async def decide(request, kit):
    found = await kit.recall(request, k=1)
    if len(found) == 0:
        return {"handoff": "nothing learned applies"}
    return {"serve": found[0]["name"]}
"""


async def learn(lab):
    if len(lab.memories) > 0:
        for memory in lab.memories:
            lab.remember(**memory)
        return ROUTER
    shown = "\n\n".join(lab.render(e) for e in lab.episodes)
    reply = await lab.ask(lab.fill(KINDS_PROMPT, contract=lab.contract, episodes=shown))
    for kind in lab.parse(reply):
        lab.remember(kind["name"], kind["description"], kind["guidance"], route="none", read_only=not kind["acts"])
    return ROUTER
