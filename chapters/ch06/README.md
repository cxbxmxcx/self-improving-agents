# Chapter 6: Metacognition, learning to think less

Chapter 6 runs a metacognitive loop of monitoring and control around the calendar assistant. The harness starts with the memory and the learner it already has: the function chapter 4's GEPA search found, and chapter 5's value learning. Listing 6.1 measures what good is and how much reasoning the agent needs with and without that memory. Listing 6.2 runs the loop as conditions change, and listing 6.3 gives the loop a task its starting learner cannot learn, so the loop searches the bank of learners helix defines.

## Code listings: the reasoning floor, the online loop, and the learner search

Use the project environment described in the [repository setup](../../README.md#setup), then run the commands below from the repository root. All three listings replay saved model replies by default and require no API key. Run them in order, since listings 6.2 and 6.3 read the floors listing 6.1 writes.

```powershell
python chapters/ch06/01_reasoning_floor.py --repeats 3
python chapters/ch06/02_online_loop.py
python chapters/ch06/03_learner_search.py --promote --paired
```

| Listing | Source | What to inspect |
|---|---|---|
| 6.1, section 6.2.4 | [01_reasoning_floor.py](01_reasoning_floor.py) | Success, reasoning tokens, API cost, and model-call time with and without the starting memory, then how often three runs of each request make the same booking |
| 6.2, section 6.3.4 | [02_online_loop.py](02_online_loop.py) | Twenty requests through the recorder, controller, detector, learner, and gate, with Dana's note changing before request 7 |
| 6.3, section 6.4.3 | [03_learner_search.py](03_learner_search.py) | Requests after four colleagues add rules to their calendar notes: quarantine, two gate refusals, the learner monitor, a search of the bank scored by the gate, promotion, and the promoted memory on 24 test requests |

Replay executes the local harness and functions while serving model replies from `assistant/recorded/`. It reports the saved token usage and model-call durations, so its timing columns describe the recorded calls rather than local replay time. Repeating a replay checks reproducibility of that execution path; it does not provide a fresh model trial.

Add `--record` to make paid model calls for requests missing from the chosen recording. A new prompt, tool schema, setting, or tool result can require additional replies, and changing recorded prompt text can therefore break replay. Listings 6.1 and 6.3 take `--budget`, the dollars of live calls a run may add; listing 6.3 refuses a live call that could cross it.

## Measuring good: the reasoning floor and consistency

The reasoning floor records the lowest tested setting that meets a success threshold, making task success the measure of good and effort above the target a remaining gap. A solid floor requires all tasks to pass, while an adequate floor permits one miss in eight, requiring 14 successes in these 16-task sets. An unreached floor means none of the tested settings met the threshold, and `ReasoningFloorSignal` reports the gap against the target through `GapMeasurement`.

| Memory | none | low | medium | high | Solid floor | Adequate floor |
|---|---:|---:|---:|---:|---|---|
| No memory | 2/16 | 4/16 | 12/16 | 13/16 | Not reached | Not reached |
| Starting memory | 16/16 | 16/16 | 16/16 | 16/16 | none | none |

With the starting memory, a request costs $0.0021 at none, against $0.0162 at high without memory. These execution costs exclude acquiring and validating the memory, and the none setting still involves model inference and function execution.

With `--repeats 3`, listing 6.1 serves each development request twice more at the setting the controller picks for each arm. A recording replays one reply per request, so each repeat is kept in its own file, `reasoning_floor.repeat2.jsonl` and `reasoning_floor.repeat3.jsonl`.

| Arm and setting | Same booking all three times | Accepted all three times | Runs accepted |
|---|---:|---:|---:|
| No memory, high | 9/16 | 11/16 | 43/48 |
| Starting memory, none | 16/16 | 16/16 | 48/48 |

Recording the repeats cost $0.64.

## The online metacognitive loop

Listing 6.2 runs the parts in [online.py](assistant/online.py) around the same `run_task` call the other listings use. The controller looks up its setting from listing 6.1's floors, the detector quarantines the memory after two rejections in the last four runs that used it, and the starting learner relearns from the runs recorded since quarantine. The gate installs a relearned function only when it passes the [drift tasks'](assistant/generator_drift.py) six held-back cases.

| Requests | Memory | Setting | Accepted | Per request |
|---|---|---|---:|---:|
| 1 to 6, familiar requests | Starting memory, trusted | none | 6/6 | $0.0021 |
| 7 and 8, after Dana's note changes | Starting memory, trusted | none | 0/2 | $0.0020 |
| 9 to 12 | Quarantined | high | 3/4 | $0.0258 |
| 13 to 20 | Relearned, trusted | none | 8/8 | $0.0022 |

Relearning took three learning calls and $0.18, and recording the whole stream cost $0.32. The relearned function reads "half an hour" literally, so another rewording would trip the detector again.

## Searching the bank of learners

In listing 6.3, four colleagues the assistant already books add rules to their calendar notes: Leo takes no meetings before 10:00, Priya keeps 12:00 to 13:00 free for lunch, Omar keeps Wednesdays free, and Sam is offline from 15:30 on Tuesdays and Thursdays. Their [requests](assistant/generator_rules.py) recall the starting memory, which knows none of these rules. The [learner monitor](assistant/online.py) counts the gate's refusals in a row, and after two it schedules a search of the bank at the next learning point.

| Requests | Memory | Setting | Accepted | Reasoning tokens | Per request |
|---|---|---|---:|---:|---:|
| 1 and 2, after the rules appear | Starting memory, trusted | none | 0/2 | 0 | $0.0024 |
| 3 to 14 | Quarantined | high | 9/12 | 2,008 | $0.0126 |
| 15 to 18 | Promoted, trusted | none | 4/4 | 0 | $0.0023 |

| Learning step | Runs | Gate | Learning calls | Dollars |
|---|---:|---:|---:|---:|
| Value learning, first attempt | 4 | 3/6 | 3 | $0.2713 |
| Value learning, second attempt | 8 | 3/6 | 6 | $0.5718 |
| Search: value learning | 12 | 2/6 | 9 | $0.7783 |
| Search: reflection | 12 | 1/6 | 13 | $1.3745 |
| Search: success distillation, kept | 12 | 6/6 | 1 | $0.1375 |
| Search: transition distillation | 12 | 0/6 | 1 | $0.1177 |
| Search: skill synthesis | 12 | 5/6 | 3 | $0.1831 |

`CandidateSearch` keeps the starting learner as the incumbent and replaces it only with a strictly higher gate score. Promotion stays a person's choice: without `--promote`, the stream stops after the search reports what it would keep. With `--promote`, the promoted memory serves the 24 test requests, which no learner or gate saw, at none: 24 of 24 accepted, with 0 reasoning tokens, $0.0024, and 2.9 seconds a request.

With `--paired`, the same 24 test requests run again at high without memory: 13 of 24 accepted, with 2,561 reasoning tokens, $0.0151, and 15.9 seconds a request. Learning cost $3.43 in all, so at that saving of about $0.013 a request the full search repays after about 270 requests, and success distillation alone ($0.14) after about 11. The recording reuses the 58 calls of the pilot that designed this task ($2.39) and added 99 calls for the listing ($0.33) and 72 for the paired runs ($0.36).

## Supporting scripts

These scripts are not chapter listings. They show where the starting memory came from and keep the comparisons chapter 7 builds on.

| Script | What it does |
|---|---|
| [starting_memory_search.py](starting_memory_search.py) | Chapter 4's GEPA searching the function in procedural memory; its `--promote` wrote the starting memory |
| [learner_comparison.py](learner_comparison.py) | The five learners and a control compared offline by how closely their functions' outputs match the starting memory's |
| [generalization.py](generalization.py) | The same learner writing a function for a second task class, then the agent using one memory that holds both |
| [revenue_calibration.py](revenue_calibration.py) | An earlier calibration study kept with its recording |

The starting memory search began from the [contract-only function](assistant/memories/find_shared_starts.contract_only.py), which the user accepts on no development request. Its [evaluator](assistant/context_search.py) books each candidate's earliest start, calls every request with days written both as `wed` and as `Wednesday`, and shows the proposer the function's own output for the first rejected request.

| Recorded attempt | Change | Result | Cost |
|---|---|---|---:|
| [No proposer reasoning](assistant/recorded/context_search.no_reasoning.jsonl) | Proposer at its default setting | 0 of 16 for every candidate | $0.10 |
| [Contract-form days](assistant/recorded/context_search.contract_days.jsonl) | Proposer at medium reasoning | 16 of 16, but the agent wrote `Wednesday` and higher settings solved 15, 11, and 13 of 16 | $0.59 |
| [Terse feedback](assistant/recorded/context_search.terse_feedback.jsonl) | Evaluator calls each request both ways | Stalled at 2 of 16 | $0.63 |
| [Kept search](assistant/recorded/context_search.jsonl) | Feedback shows the function's output | v15 reaches 16 of 16; 16 of 16 on validation and test | $0.90 |

The kept function matches the [hand-written version](assistant/memories/find_shared_starts.py) of the same contract on all 48 development, validation, and test cases, measured by `BehaviorMatchSignal`. The hand-written function stays in the repository as a readable check and as the optional `hand_written` arm of listing 6.1.

## Following experience into memory

| Component | Role |
|---|---|
| [generator.py](assistant/generator.py), [generator_same_time.py](assistant/generator_same_time.py), [generator_drift.py](assistant/generator_drift.py), [generator_rules.py](assistant/generator_rules.py) | Generate fresh calendars and requests from fixed seeds, retaining solvable cases that expose the conventions, the changed note, or the colleagues' rules |
| [environment.py](assistant/environment.py), [tools.py](assistant/tools.py) | Simulate calendar reads, bookings, cancellations, and final outcome checks |
| [find_shared_starts.searched.py](assistant/memories/find_shared_starts.searched.py) | The starting memory for individual meetings on requested days |
| [find_shared_starts.team_rules.py](assistant/memories/find_shared_starts.team_rules.py) | A hand-written check that honors the colleagues' rules; never offered to the agent |
| [find_common_starts.py](assistant/memories/find_common_starts.py) | Reference starts that work at the same time across all requested days |
| [harness.py](assistant/harness.py) | Build the assistant with procedural memory, callable functions, and an effort setting; name the starting memory and learner |
| [context_search.py](assistant/context_search.py) | The evaluator, validation hook, and prompts that let GEPA search code |
| [learning.py](assistant/learning.py) | Build episodes and contracts, provide practice, and compare outputs |
| [online.py](assistant/online.py) | The recorder, controller, detector, gate, and learner monitor of the online loop |
| [learning_mechanisms.py](../../helix/memory/learning_mechanisms.py) | The five learners in the bank and the no-experience control |
| [recording.py](assistant/recording.py) | Serve recorded replies or append missing live replies |

Context injection and tool construction call the same retrieval helper separately, with matching request, user context, and `k`. The chapter holds procedural memory fixed during each run so these paths select the same entries. The [code-skill guard](../../helix/memory/code_skills.py) checks source and limits execution, but does not provide a security sandbox.

| Task class | Seeds |
|---|---|
| Multi-attendee | development 1–16, validation 101–116, test 201–216 |
| Same-time | development 301–316, validation 401–416, test 501–516 |
| Drift (listing 6.2) | stream 701–720, gate 801–806 |
| Colleagues' rules (listing 6.3) | first two requests 17–18, stream 1–16, gate 801–806, test 201–224 |

## Keeping the earlier experiments

The earlier task designs preserve the evidence used to choose the calendar conventions and callable memory representation. Their results belong to different environments or prompts and should be read as development history.

| Earlier design | Files or recordings | Observed result |
|---|---|---|
| Text procedure on fresh weeks | [multi_attendee_scheduling.md](assistant/memories/multi_attendee_scheduling.md), `reasoning_floor.fresh_weeks.jsonl` | 0, 3, 5, 6 successes without memory; 1, 5, 6, 7 with text, over 8 tasks |
| Recurring team | [generator_recurring_team.py](assistant/generator_recurring_team.py), [team_windows.recurring_team.md](assistant/memories/team_windows.recurring_team.md) | Little room to improve a baseline already solving 15 of 16 tasks at low |
| Plain calendars | [environment_plain_calendars.py](assistant/environment_plain_calendars.py), [generator_plain_calendars.py](assistant/generator_plain_calendars.py), [reference function](assistant/memories/find_shared_starts.plain_calendars.py) | Callable memory solved all 16 at none, but the contract already described the method |
| Ambiguous client note | `reasoning_floor.ambiguous_note.jsonl`, `learning_search.ambiguous_note.jsonl` | Exposed misleading crash feedback and an unresolved boundary rule |
| Hand-written starting memory | [find_shared_starts.py](assistant/memories/find_shared_starts.py), [floors](runs/reasoning_floor.hand_written.json) | 16 of 16 at every setting; earlier versions of the chapter used it as the reference |
| Decision-model scaffold | [Legacy appendix](../../chapter_appendices/ch06_decision_scaffold_legacy/README.md) | Preserves the retired chapter design |
