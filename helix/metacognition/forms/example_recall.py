"""Example recall: accepted runs shown to the agent as worked examples. No writer; practice picks the set."""

HEADER = """Worked examples of this job from earlier requests, each with the tool
calls that were made and the answer that was accepted:"""
K = 3
PRACTICE = 6


async def learn(lab):
    newest = []
    seen = set()
    for e in reversed(lab.episodes):
        if e.success and e.task_id not in seen:
            seen.add(e.task_id)
            newest.append(e)
    if not newest:
        return HEADER + "\n\n(no accepted runs yet)"
    candidates = [newest[:K], newest[K:2 * K], newest[::2][:K]]
    best = None
    best_score = -1.0
    for chosen in candidates:
        if not chosen:
            continue
        shown = set(e.task_id for e in chosen)
        text = HEADER + "\n\n" + "\n\n".join(lab.render(e) for e in chosen)
        held_out = [t for t in lab.stream() if t not in shown][:PRACTICE]
        score = 0.0
        for task_id in held_out:
            reward, reason, observed = await lab.practice(text, task_id)
            score += reward
        if held_out:
            score = score / len(held_out)
        if score > best_score:
            best = text
            best_score = score
    return best
