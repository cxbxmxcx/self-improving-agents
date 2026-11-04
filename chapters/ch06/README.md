# Chapter 6: Metacognition, learning to think less

Chapter 6 examines metacognition as thinking about thinking, with learning to think less measured through task success and reasoning effort. The current manuscript, [draft 0.10](chapter6.draft0.10.md), cut to 31 Word pages for the 35-page maximum, follows one pattern: chapter 4's context search finds the procedural memory that defines what good looks like, a search over learning mechanisms finds one that reproduces it from experience, and the kept learner runs online in a metacognition loop. [Editable figures and previews](figures/README.md) support the explanations, and the [code plan](../plan_ch06_code.md) and the review notes ([continuity](../../docs/editorial/ch06-review-0.6.md), [figure style](../../docs/editorial/ch06-review-0.7.md), [content](../../docs/editorial/ch06-content-review-0.7.md)) record how earlier drafts developed.

The [Word chapter](../../06_Lanham5_Self-Improving_Agents.docx) contains draft 0.10 in the existing Manning template, and the [versioned Word copy](../../06_Lanham5_Self-Improving_Agents.draft0.10.docx) contains the same text, tables, listings, and eleven figures. The previous Word documents are preserved as [draft 0.9](../../06_Lanham5_Self-Improving_Agents.draft0.9.docx), [draft 0.8](../../06_Lanham5_Self-Improving_Agents.draft0.8.docx), [draft 0.7](../../06_Lanham5_Self-Improving_Agents.draft0.7.docx), [draft 0.6](../../06_Lanham5_Self-Improving_Agents.draft0.6.docx), and [draft 0.5](../../06_Lanham5_Self-Improving_Agents.draft0.5.docx).

## Code listings: context search, reasoning floor, learning search, generalization, and the online loop

Use the project environment described in the [repository setup](../../README.md#setup), then run the commands below from the repository root. All five listings replay saved model replies by default and require no API key. Run them in order, since listing 6.4 loads the function listing 6.3 learned and listing 6.5 starts from the mechanism listing 6.3 kept.

```powershell
python chapters/ch06/01_context_search.py
python chapters/ch06/02_reasoning_floor.py
python chapters/ch06/03_learning_search.py
python chapters/ch06/04_generalization.py
python chapters/ch06/05_online_loop.py
```

| Listing | Source | What to inspect |
|---|---|---|
| 6.1, section 6.2.2 | [01_context_search.py](01_context_search.py) | GEPA searching the procedural-memory function: each version's parent, operator, and accepted requests, then the kept version's validation and test results |
| 6.2, section 6.2.3 | [02_reasoning_floor.py](02_reasoning_floor.py) | Success, reasoning tokens, API cost, and model-call time with and without the bootstrapped memory |
| 6.3, section 6.3.3 | [03_learning_search.py](03_learning_search.py) | Validation match for five mechanisms and a control, followed by the selected function's test match and agent floor |
| 6.4, section 6.4.1 | [04_generalization.py](04_generalization.py) | The unchanged mechanism learning a second function, then the agent using one memory containing both functions |
| 6.5, section 6.5.2 | [05_online_loop.py](05_online_loop.py) | Twenty requests through the recorder, detector, learner, gate, and controller, with Dana's note changing before request 7 |

Replay executes the local harness and functions while serving model replies from `assistant/recorded/`. It reports the saved token usage and model-call durations, so its timing columns describe the recorded calls rather than local replay time. Repeating a replay checks reproducibility of that execution path; it does not provide a fresh model trial.

Add `--record` to make paid model calls for requests missing from the chosen recording. A new prompt, tool schema, setting, or tool result can require additional replies, and changing recorded prompt text can therefore break replay. Listing 6.1 gives GEPA its own seeded generator, because live calls can draw from Python's module-level `random` and a replay must choose parents as the recorded search did.

## Context search and the bootstrapped memory

Listing 6.1 searches the function in procedural memory the way chapters 2 to 4 searched prompts, starting from the [contract-only function](assistant/memories/find_shared_starts.contract_only.py) that the user accepts on no development request. Its [evaluator](assistant/context_search.py) books each candidate's earliest start, calls every request with days written both as `wed` and as `Wednesday`, and shows the proposer the function's own output for the first rejected request. A person promotes the kept version with `--promote`, which writes [find_shared_starts.searched.py](assistant/memories/find_shared_starts.searched.py), the bootstrapped memory every later listing uses.

| Recorded attempt | Change | Result | Cost |
|---|---|---|---:|
| [No proposer reasoning](assistant/recorded/context_search.no_reasoning.jsonl) | Proposer at its default setting | 0 of 16 for every candidate | $0.10 |
| [Contract-form days](assistant/recorded/context_search.contract_days.jsonl) | Proposer at medium reasoning | 16 of 16, but the agent wrote `Wednesday` and higher settings solved 15, 11, and 13 of 16 | $0.59 |
| [Terse feedback](assistant/recorded/context_search.terse_feedback.jsonl) | Evaluator calls each request both ways | Stalled at 2 of 16 | $0.63 |
| [Kept search](assistant/recorded/context_search.jsonl) | Feedback shows the function's output | v15 reaches 16 of 16; 16 of 16 on validation and test | $0.90 |

The kept function matches the [hand-written version](assistant/memories/find_shared_starts.py) of the same contract on all 48 development, validation, and test cases. The hand-written function stays in the repository as a readable check and as the optional `hand_written` arm of listing 6.2.

## Reasoning floor and behavior-match signals

The reasoning floor records the lowest tested setting that meets a success threshold, making task success the measure of good and effort above the target a remaining gap. A solid floor requires all tasks to pass, while an adequate floor permits one miss in eight, requiring 14 successes in these 16-task sets. An unreached floor means none of the tested settings met the threshold, and `ReasoningFloorSignal` reports the gap against the target through `GapMeasurement`.

| Task class and memory | none | low | medium | high | Solid floor | Adequate floor |
|---|---:|---:|---:|---:|---|---|
| Multi-attendee, no memory | 2/16 | 4/16 | 12/16 | 13/16 | Not reached | Not reached |
| Multi-attendee, bootstrapped memory | 16/16 | 16/16 | 16/16 | 16/16 | none | none |
| Same-time, no memory | 4/16 | 2/16 | 10/16 | 14/16 | Not reached | high |
| Same-time, bootstrapped memory | 16/16 | 16/16 | 16/16 | 16/16 | none | none |

From the table, the bootstrapped functions support all measured development tasks at none. The multi-attendee runs cost $0.0021 per task with that memory at none, compared with $0.0162 at high without memory; the same-time comparison is $0.0028 against $0.0215. These execution costs exclude acquiring and validating the memory, and the none setting still involves model inference and function execution.

| Mechanism | Multi-attendee validation match | Writer calls |
|---|---:|---:|
| No experience, control | 0/16 | 1 |
| Value learning | 16/16 | 16 |
| Reflection | 16/16 | 7 |
| Success distillation | 10/16 | 1 |
| Transition distillation | 16/16 | 1 |
| Skill synthesis | 16/16 | 1 |

`BehaviorMatchSignal` measures the fraction of complete outputs that match the reference, and candidate search keeps value learning as the first mechanism to reach the highest validation score. Its function matches all 16 multi-attendee test cases and supports all 16 validation tasks at none, while the same mechanism learns a separate function that achieves those results for the same-time class. With both learned functions available at `k=2`, the agent solves all 16 validation tasks of each class at none.

## The online metacognition loop

Listing 6.5 runs the parts in [online.py](assistant/online.py) around the same `run_task` call the other listings use. The controller looks up its setting from listing 6.2's floors, the detector quarantines the memory after two rejections in the last four runs that used it, and the learner relearns from the runs recorded since quarantine. The gate installs a relearned function only when it passes the [drift tasks'](assistant/generator_drift.py) six held-back cases.

| Requests | Memory | Setting | Accepted | Per request |
|---|---|---|---:|---:|
| 1 to 6, first class | Learned, trusted | none | 6/6 | $0.0021 |
| 7 and 8, after Dana's note changes | Learned, trusted | none | 0/2 | $0.0020 |
| 9 to 12 | Quarantined | high | 3/4 | $0.0258 |
| 13 to 20 | Relearned, trusted | none | 8/8 | $0.0022 |

Relearning took three writer calls and $0.18, and recording the whole stream cost $0.32. The relearned function reads "half an hour" literally, so another rewording would trip the detector again; finding a better learner for such a task class is the offline loop's job.

## Following experience into memory

Each class contributes 32 development episodes from medium- and high-effort runs without memory. The first class has 25 accepted runs and the second has 24, with acceptance supplied by the simulated checker. Episodes retain requests, tool interactions, and outcomes, while the learning mechanisms produce callable procedural memory.

| Component | Role |
|---|---|
| [generator.py](assistant/generator.py), [generator_same_time.py](assistant/generator_same_time.py), [generator_drift.py](assistant/generator_drift.py) | Generate fresh calendars and requests from fixed seeds, retaining solvable cases that expose the conventions or the changed note |
| [environment.py](assistant/environment.py), [tools.py](assistant/tools.py) | Simulate calendar reads, bookings, cancellations, and final outcome checks |
| [find_shared_starts.searched.py](assistant/memories/find_shared_starts.searched.py) | The bootstrapped memory for individual meetings on requested days, found by listing 6.1 |
| [find_common_starts.py](assistant/memories/find_common_starts.py) | Reference starts that work at the same time across all requested days |
| [harness.py](assistant/harness.py) | Build the assistant with procedural memory, callable functions, and an effort setting |
| [context_search.py](assistant/context_search.py) | The evaluator, validation hook, and prompts that let GEPA search code |
| [learning.py](assistant/learning.py) | Build episodes and contracts, provide development practice, and compare outputs |
| [online.py](assistant/online.py) | The recorder, controller, detector, and gate of the online loop |
| [learning_mechanisms.py](../../helix/memory/learning_mechanisms.py) | Implement the five mechanisms and the no-experience control |
| [recording.py](assistant/recording.py) | Serve recorded replies or append missing live replies |

Context injection and tool construction call the same retrieval helper separately, with matching request, user context, and `k`. The chapter holds procedural memory fixed during each run so these paths select the same entries. The [code-skill guard](../../helix/memory/code_skills.py) checks source and limits execution, but does not provide a security sandbox.

| Task class | Development seeds | Validation seeds | Test seeds |
|---|---|---|---|
| Multi-attendee | 1–16 | 101–116 | 201–216 |
| Same-time | 301–316 | 401–416 | 501–516 |
| Drift (online stream and gate) | | | stream 701–720, gate 801–806 |

## Reviewing and promoting a result

Listing 6.1 writes the kept source to `runs/context_search/`, and `--promote` writes the bootstrapped memory. Listing 6.3 writes the selected learner's source and measurements into `runs/learning_search/`, and its `--promote` writes `assistant/memories/learned_find_shared_starts.py`, which listing 6.4 prefers when loading the first class's memory. Neither flag enforces a score threshold, so inspect validation, test, and agent results before using it.

Without a promoted learned file, listing 6.4 tries the learned source saved by listing 6.3. If neither source is present, it uses the bootstrapped first-class function and labels that fallback in its output. Listing 6.5 always starts from the learned source in `runs/learning_search/` and writes its relearned function to `runs/online_loop/`.

## Keeping the earlier experiments

The earlier task designs preserve the evidence used to choose the calendar conventions and callable memory representation. Their results belong to different environments or prompts and should be read as development history. Drafts [0.1](chapter6.draft0.1.md) to [0.9](chapter6.draft0.9.md) retain their original prose and figure references.

| Earlier design | Files or recordings | Observed result |
|---|---|---|
| Text procedure on fresh weeks | [multi_attendee_scheduling.md](assistant/memories/multi_attendee_scheduling.md), `reasoning_floor.fresh_weeks.jsonl` | 0, 3, 5, 6 successes without memory; 1, 5, 6, 7 with text, over 8 tasks |
| Recurring team | [generator_recurring_team.py](assistant/generator_recurring_team.py), [team_windows.recurring_team.md](assistant/memories/team_windows.recurring_team.md) | Little room to improve a baseline already solving 15 of 16 tasks at low |
| Plain calendars | [environment_plain_calendars.py](assistant/environment_plain_calendars.py), [generator_plain_calendars.py](assistant/generator_plain_calendars.py), [reference function](assistant/memories/find_shared_starts.plain_calendars.py) | Callable memory solved all 16 at none, but the contract already described the method |
| Ambiguous client note | `reasoning_floor.ambiguous_note.jsonl`, `learning_search.ambiguous_note.jsonl` | Exposed misleading crash feedback and an unresolved boundary rule |
| Hand-written bootstrapped memory | [find_shared_starts.py](assistant/memories/find_shared_starts.py), [floors](runs/reasoning_floor.hand_written.json) | 16 of 16 at every setting; drafts 0.1 to 0.8 used it as the reference |
| Decision-model scaffold | [Legacy appendix](../../chapter_appendices/ch06_decision_scaffold_legacy/README.md) | Preserves the retired chapter design |
