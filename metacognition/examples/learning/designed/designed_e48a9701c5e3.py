"""Robust transition distillation: learn from all runs, but trust successes and use failures only as counterexamples."""

PROMPT = """{contract}

You are writing one Python function for this contract.

The only authoritative evidence is the successful runs. Failed runs are counterexamples only: they can rule out actions, but they do not define the intended behavior.

Observed runs:
{episodes}

Writing rules:
- Match the successful behavior as closely as possible.
- Infer the smallest rule set that explains the successful runs.
- If a capability appears only in failures, do not use it.
- If a capability appears in a successful run but may be unavailable in general, guard it with hasattr/getattr and a safe fallback.
- If the requests or traces show an explicit special case, encode that branch directly instead of averaging across examples.
- Keep the function generic to the contract; do not invent hidden APIs, optional methods, or domain shortcuts.
- Return only the function source.
"""

def episode_text(lab, episode):
    status = "SUCCESS" if episode.success else f"FAILURE: {episode.reason}"
    return (
        f"### Episode {episode.task_id} [{status}]\n"
        f"Request:\n{episode.request}\n\n"
        f"{lab.render(episode)}"
    )

def unique_failures(episodes, limit=6):
    picked = []
    seen = set()
    for episode in episodes:
        if episode.success:
            continue
        reason = episode.reason or ""
        key = reason[:120]
        if key in seen:
            continue
        seen.add(key)
        picked.append(episode)
        if len(picked) >= limit:
            break
    return picked

async def learn(lab):
    successes = [episode for episode in lab.episodes if episode.success]
    failures = unique_failures(lab.episodes, limit=6)

    parts = []
    parts.append(f"### Successful runs ({len(successes)})")
    if successes:
        parts.extend(episode_text(lab, episode) for episode in successes)
    else:
        parts.append("None")

    parts.append(f"### Counterexamples ({len([episode for episode in lab.episodes if not episode.success])})")
    if failures:
        parts.extend(episode_text(lab, episode) for episode in failures)
    else:
        parts.append("None")

    prompt = lab.fill(PROMPT, contract=lab.contract, episodes="\n\n".join(parts))
    return await lab.write(prompt)