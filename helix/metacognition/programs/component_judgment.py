"""Memories by kind of request: a decision program for each kind that acts, read-only guidance for each kind that doesn't.

A hand-written control. Jev recognizes the kind of request from the memories' descriptions, so what a request is asking
for stays a judgment, and each decision program only carries out the one kind of work it was written for.
"""

KINDS_PROMPT = """{contract}

Below are runs where the assistant, a language model, handled requests for
this job: each request, every tool call with its result, and the verdict.
Some requests are worded oddly, some should not be acted on at all, and
some runs went wrong.

List the kinds of request these runs show: one kind for each action the job
can take, and as few others as cover the requests that should not be acted
on. A request can use all of a job's own words while asking whether it was
done, telling the assistant not to do it, asking only for information, or
asking to undo or change it. Those are not the job's kinds, so say so in the
descriptions.

Reply with only a JSON list. Each item has the keys "name" (short, in
snake_case), "description" (which requests belong to this kind, and which
similar ones don't), "guidance" (what the assistant should know about this
kind), and "acts" (true if a request of this kind should end in an action,
false if it should only be answered or passed on).

{episodes}"""

PROGRAM_PROMPT = """{contract}

Below are runs where the assistant, a language model, handled requests for
this job. Write a decision program that serves one kind of request, with no
language model:

{name}: {description}

A request reaches this program only after it has been recognized as this
kind, so don't judge again what it asks for. Code finds the ids in a request,
whatever form they come in, works out from these runs how ids are really
written in the tools, and calls the procedures and the action. When a request
names several ids, ask Jev which one it means. Hand off anything you can't
finish: a wrong action costs far more than a hand-off.

{episodes}"""

REPAIR_PROMPT = """These are the kinds of request an assistant knows, by name and description:

{kinds}

Each request is matched to the kind whose description fits it best. These
requests came out wrong:

{failures}

Rewrite the descriptions so that each request matches the right kind and the
others still do. Reply with only a JSON object from each kind's name to its
new description."""

ROUTER = """async def decide(request, kit):
    found = await kit.recall(request, k=8)
    if len(found) == 0:
        return {"handoff": "nothing learned applies"}
    criteria = {}
    for memory in found:
        criteria[memory["name"]] = memory["description"]
    question = {"type": "choice", "criteria": criteria, "instructions": (
        "Which kind of request is `request`? Go by what it actually asks for now. A request can use a job's own "
        "words while asking whether it was done, telling the assistant not to do it, asking only for information, "
        "or asking to undo or change it.")}
    answers = await kit.ask({"request": request}, {"kind": question})
    return {"serve": answers["kind"]["choice"]}
"""

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
    kinds = lab.parse(await lab.ask(lab.fill(KINDS_PROMPT, contract=lab.contract, episodes=shown)))
    programs = {}
    for kind in kinds:
        if kind["acts"]:
            prompt = lab.fill(PROGRAM_PROMPT, contract=lab.contract, name=kind["name"], description=kind["description"],
                              episodes=shown)
            programs[kind["name"]] = code_of(await lab.ask(prompt))

    def store():
        for kind in kinds:
            if kind["name"] in programs:
                lab.remember(kind["name"], kind["description"], programs[kind["name"]], route="program")
            else:
                lab.remember(kind["name"], kind["description"], kind["guidance"], route="none", read_only=True)

    store()
    for attempt in range(ROUNDS):
        failures = []
        score = 0.0
        for task_id in lab.stream():
            reward, reason, served_by = await lab.practice(ROUTER, task_id)
            score += reward
            if reward < 1:
                failures.append("Request: " + lab.request(task_id) + "\nOutcome: " + reason + "\nMatched: " + served_by)
        lab.note("attempt " + str(attempt) + ": score " + str(score))
        if not failures or attempt == ROUNDS - 1:
            break
        listing = "\n".join(kind["name"] + ": " + kind["description"] for kind in kinds)
        shown_failures = "\n\n".join(failures[:SHOWN_FAILURES])
        changes = lab.parse(await lab.ask(lab.fill(REPAIR_PROMPT, kinds=listing, failures=shown_failures)))
        for kind in kinds:
            kind["description"] = changes.get(kind["name"], kind["description"])
        store()
    return ROUTER
