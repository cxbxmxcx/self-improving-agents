"""Contract-grounded synthesis with reflective repair after runtime checking."""

DRAFTPROMPT = """{contract}

Below are the judged successful runs:

{wins}

Below are the judged unsuccessful runs:

{losses}

Write one Python function that solves the job.

Rules:
- Use only names and behaviors that are explicit in the contract or clearly evidenced by successful runs.
- Do not invent tool or method names from the natural-language request.
- If a capability is not clearly available, prefer a safe fallback over crashing.
- Guard external calls so missing attributes do not raise AttributeError.
- If the contract gives a policy or rule, implement that policy directly instead of replacing it with a generic refusal.
- Keep the solution general; it must work on held-out cases and should not assume a specific domain.
- Return only the function source.
"""

REPAIRPROMPT = """{contract}

The previous draft was tested and failed on these cases:

{failures}

Here are checks it already handled successfully:

{successes}

Current source:

{source}

Revise it into a corrected single Python function.

Repair rules:
- Fix every crash and obvious hallucinated API name.
- Preserve behavior that already worked.
- Use only exact contract names or names directly observed in successful traces.
- If the environment lacks a capability you expected, handle it safely instead of calling a non-existent method.
- If a case was declined or refused even though the contract or policy text actually permits it, correct the logic so the policy is honored.
- Do not add commentary; return only the function source.
"""

SECONDPROMPT = """{contract}

The latest draft still failed on these cases:

{failures}

These checks still worked:

{successes}

Current source:

{source}

Make one more correction pass on the same single Python function.

Repair rules:
- Prefer the contract over any earlier guess.
- Remove blanket refusals unless the contract explicitly requires them.
- If a policy says a request should be accepted, implement that acceptance path.
- Preserve all behavior that passed.
- Return only code, no explanation.
"""

FRESHPROMPT = """{contract}

A previous attempt still missed some checked cases.

Failed cases:

{failures}

Working cases:

{successes}

Current source:

{source}

Write a fresh single Python function from scratch, using the contract and the evidence above.

Rules:
- Do not rely on earlier source structure if it seems to be causing the miss.
- Preserve any behavior that is clearly correct from the working cases.
- If the contract contains a policy or exception, follow it literally.
- Avoid unsafe crashes from missing attributes or unsupported operations.
- Return only the function source.
"""


def jointext(parts):
    return "\n\n".join(parts)


def formatepisode(episode_text, request_text):
    return f"request: {request_text}\n\n{episode_text}"


def formatresults(results, keepgood, lab=None):
    parts = []
    for item in results:
        taskid, reward, reason, observations = item
        if keepgood:
            if reward <= 0:
                continue
        else:
            if reward > 0:
                continue
        request_text = ""
        if lab is not None:
            request_text = lab.request(taskid)
        parts.append(
            f"task id: {taskid}\n"
            f"request: {request_text}\n"
            f"reward: {reward}\n"
            f"reason: {reason}\n"
            f"observations: {observations}"
        )
    return jointext(parts)


def scorepair(results):
    accepted = 0
    total = 0
    for item in results:
        reward = item[1]
        total += reward
        if reward > 0:
            accepted += 1
    return accepted, total


def hasfailures(results):
    for item in results:
        reward = item[1]
        reason = str(item[2])
        if reward <= 0:
            return True
        if "AttributeError" in reason:
            return True
        if "NameError" in reason:
            return True
        if "TypeError" in reason:
            return True
        if "crash" in reason.lower():
            return True
    return False


async def checksource(lab, source):
    checked = []
    seen = set()
    for taskid in lab.stream():
        if taskid in seen:
            continue
        seen.add(taskid)
        reward, reason, observations = await lab.practice(source, taskid)
        checked.append((taskid, reward, reason, observations))
    return checked


async def learn(lab):
    wins = []
    losses = []
    for episode in lab.episodes:
        text = lab.render(episode)
        if episode.success:
            wins.append(formatepisode(text, lab.request(episode.task_id)))
        else:
            losses.append(formatepisode(text, lab.request(episode.task_id)))

    draft = await lab.write(
        lab.fill(
            DRAFTPROMPT,
            contract=lab.contract,
            wins=jointext(wins),
            losses=jointext(losses),
        )
    )

    candidates = []
    firstcheck = await checksource(lab, draft)
    candidates.append((draft, firstcheck))

    if hasfailures(firstcheck):
        repair1 = await lab.write(
            lab.fill(
                REPAIRPROMPT,
                contract=lab.contract,
                failures=formatresults(firstcheck, False, lab),
                successes=formatresults(firstcheck, True, lab),
                source=draft,
            )
        )
        secondcheck = await checksource(lab, repair1)
        candidates.append((repair1, secondcheck))

        if hasfailures(secondcheck):
            repair2 = await lab.write(
                lab.fill(
                    SECONDPROMPT,
                    contract=lab.contract,
                    failures=formatresults(secondcheck, False, lab),
                    successes=formatresults(secondcheck, True, lab),
                    source=repair1,
                )
            )
            thirdcheck = await checksource(lab, repair2)
            candidates.append((repair2, thirdcheck))

            if hasfailures(thirdcheck):
                fresh = await lab.write(
                    lab.fill(
                        FRESHPROMPT,
                        contract=lab.contract,
                        failures=formatresults(thirdcheck, False, lab),
                        successes=formatresults(thirdcheck, True, lab),
                        source=repair2,
                    )
                )
                fourthcheck = await checksource(lab, fresh)
                candidates.append((fresh, fourthcheck))

    bestsource = candidates[0][0]
    bestscore = scorepair(candidates[0][1])

    for source, results in candidates[1:]:
        score = scorepair(results)
        if score > bestscore:
            bestsource = source
            bestscore = score

    return bestsource