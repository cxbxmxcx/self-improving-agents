# Chapter 5: Reinforcement learning for memory improvement

A personal agent creates writing blocks in a local calendar and uses episodic experience to inform later scheduling. The chapter separates valid execution, explicit user acceptance, learned memory values, and evidence of improvement.

## Run the book's listings

Run from the repository root in the project environment. The three files follow their first appearance in the chapter. The default walkthrough uses local data with no model downloads or provider calls.

| Listing | Source | What the chapter demonstrates |
|---------|--------|-------------------------------|
| 5.1 | [01_memory_value.py](01_memory_value.py) | Outcome-based value updates and a controlled recall calculation |
| 5.2 | [02_calendar_agent.py](02_calendar_agent.py) | Calendar writes, explicit review, reward, and agent construction |
| 5.3 | [03_memory_learning.py](03_memory_learning.py) | Outcome measurement, persistent memory, learning hooks, four requests, and handoff |

```powershell
python chapters/ch05/01_memory_value.py
python chapters/ch05/02_calendar_agent.py
python chapters/ch05/03_memory_learning.py
```

The second script performs a direct calendar check in a temporary directory. The third creates a persistent run directory under `runs/personal/` containing `calendar.json`, the SQLite stores, `trace.json`, and a transcript. Use `--output PATH` to choose a new or empty directory; existing evidence is preserved.

The recorded sequence creates events at 14:00, 14:00, 14:00, and 09:00. All four are valid, three receive acceptance, and one receives a correction about the user's changed work style. The correction informs a later booking; it does not automatically move the rejected event.

```text
Accepted first bookings: 3/4 (75%); valid: 4; invalid: 0; no feedback: 0
```

These are scripted demonstration results, not a measured language-model improvement. The fourth request explicitly refers to feedback from the rejected booking without repeating the preferred time. With the fixed lexical retriever, a bare scheduling request retrieves an older accepted episode, so the chapter makes that limitation explicit.

The final script accepts `--live --model YOUR_MODEL_ID` with that provider's credentials. It then uses a real model with the same local calendar and scripted reviews, so actions and rewards may differ. Neither mode connects to a live calendar account.

## Supporting material

- [Personal calendar data and responder](personal/README.md).
- [Recorded calendar, trace, and recall audit](personal/recorded/README.md).

The learning rule remains `q_new = q + 0.5 * (credit - q)`. Equal credit, one recalled episode, utility weight `0.35`, recency weight zero, the encoder, and the prompt stay fixed. New episodes start at `0.5`; only earlier recalled entries receive the current request's credit.

The superseded studio example is preserved in [its historical appendix](../../chapter_appendices/ch05_studio_legacy/README.md). The [earlier research examples](../../chapter_appendices/ch05_research_legacy/README.md) and [separate RL primer](../../chapter_appendices/ch05_rl_primer_legacy/README.md) also remain outside the active teaching path.
