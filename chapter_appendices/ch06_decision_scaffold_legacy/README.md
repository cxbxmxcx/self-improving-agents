# Superseded: chapter 6 as a decision-model scaffold

This was chapter 6's first design, retired on 2026-10-31 when the chapter turned to learning procedural memory that moves a task from System 2 to System 1. It is kept, runnable and tested, because the framework pieces it exercises remain in Helix and its live evidence is a useful record.

The chapter 5 assistant gains a meta level. A decision model answers typed questions about each run, and a configured policy turns the answers into control: the fast path, the deliberate path, or a question back, a confirmation before a consequential call, a revision of an incomplete answer, and a lesson after a rejection. Chapter 5's memory learning keeps running underneath, unchanged.

The language model is gpt-5.4-mini on both paths. The fast path runs it without reasoning; the deliberate path runs it at high reasoning effort after the planner's checklist. The decision model is TypeSafe's Jev.

## Run the book's listings

Run from the repository root in the project environment. By default the assistant's model calls come from a scripted stand-in, and every decision and search proposal is replayed from a recording, so no key is needed.

| Listing | Source | What the chapter demonstrates |
|---------|--------|-------------------------------|
| 6.1 | [01_decision_model.py](01_decision_model.py) | Typed questions and answers, with confidence, tokens, seconds, and price per call |
| 6.2 | [02_metacognitive_assistant.py](02_metacognitive_assistant.py) | The scaffold on the assistant: routing, confirmation, revision, and a lesson that changes a later run |
| 6.3 | [03_metacognition_evals.py](03_metacognition_evals.py) | The controlled comparison on held-out requests, with calibration, interventions, and behavior diffs |
| 6.4 | [04_decision_search.py](04_decision_search.py) | Thresholds, questions, and the composite searched offline; thresholds tuned online with shadow tests |

```powershell
python chapter_appendices/ch06_decision_scaffold_legacy/01_decision_model.py
python chapter_appendices/ch06_decision_scaffold_legacy/02_metacognitive_assistant.py
python chapter_appendices/ch06_decision_scaffold_legacy/04_decision_search.py
python chapter_appendices/ch06_decision_scaffold_legacy/03_metacognition_evals.py
```

Each listing runs in about five seconds without keys. Listing 6.3 compares the searched configuration that listing 6.4 recorded, so the order above follows the data rather than the chapter.

## What the recorded runs show

These are scripted demonstration results with Jev's recorded decisions, not a measured improvement. The stand-in's mistakes are written into each request's script, so its deliberate path is perfect by construction; the live pass below measures a real model.

The walkthrough routes as the chapter describes: the simple booking goes fast, three sessions with Sam Ortiz deliberate, "my meeting with Sam" asks which Sam, and the send to Dana is held for confirmation.

```text
F01: reward 1; route fast
D01: reward 1; route deliberate (multi_event_limit 0.68)
A01: reward 1; route ask (ambiguous_reference 0.60)
C01: reward 1; route fast; send_reply confirm
D08: reward 1; route fast; revised the reply
W01: reward 0; route fast; lesson: Maya now wants writing blocks in the morning: use 09:00 for writing unless she names a time.
M01: reward 1; route fast
Decision model: 13 calls, 9,979 input tokens, $0.00042, 1.40 s recorded
```

The search rewrote the seed's two routing questions as narrow, observable properties and added a question the seed lacked, which sends a reply covering several questions to the deliberate path. It also lowered both routing thresholds from 0.5 to 0.4. The reviewer saw the held-out evidence (score 0.684 to 0.778, accepted 15 to 17 of 20, nothing unsafe) and promoted the new metacognition composite; online, four threshold nudges were shadow-tested and none was promoted.

On the 20 held-out requests the searched configuration accepted more than the seed and spent less, with no unsafe action. A generic lesson ("Keep writing blocks productive and focused.") is flagged as reflection theater, while the three real lessons change what the assistant does.

```text
condition                accepted unsafe  score  $/request s/request decisions regret
always fast                 12/20      3  0.426    0.00591      0.00       0.0  0.337
always deliberate           20/20      0  0.822    0.01782      0.00       0.0  0.100
seed configuration          15/20      0  0.684    0.00857      0.21       2.0  0.197
searched configuration      17/20      0  0.778    0.00774      0.19       1.8  0.134
LLM decision model          16/20      0  0.703    0.00941      1.24       1.8  0.095
```

The score is the outcome less what the run cost: a cent costs 0.1 points, and so do ten seconds of waiting (`WEIGHTS` in [assistant/evaluation.py](assistant/evaluation.py)). Those prices are a judgment, and the comparison depends on them.

## Recording and live runs

`--record` refreshes the recordings: the stand-in still runs the assistant, and any decision or proposal missing from the recording goes to the live model and joins it. Each question keeps one recorded answer, so every listing replays the same decisions. Jev's key goes in the repository's `.env` as `TYPESAFE_API_KEY` or `JEV_API_KEY`; the proposer and the LLM decision model use `OPENAI_API_KEY`.

`--live` runs gpt-5.4-mini for the assistant as well. Listings 6.3 and 6.4 print a cost estimate and stop unless you add `--yes`, and listing 6.3 takes `--trials N` to show spread and `--searched PATH` to compare a live search's result.

```powershell
python chapter_appendices/ch06_decision_scaffold_legacy/04_decision_search.py --live --yes
python chapter_appendices/ch06_decision_scaffold_legacy/03_metacognition_evals.py --live --yes --trials 3 --searched chapter_appendices/ch06_decision_scaffold_legacy/runs/search/<run>/searched.json
```

A live walkthrough on 2026-10-12 cost $0.035 for the model and $0.0005 for Jev. Deliberating cost about four times the fast path, and asking was cheapest. Live runs vary: in one, the model only drafted the reply to Dana, so the monitor had nothing to hold.

## The live pass

A live pass of listings 6.4 and 6.3 on 2026-10-12 cost about $2.60 and is kept in [assistant/recorded/live/](assistant/recorded/live/README.md). With gpt-5.4-mini, the seed configuration accepted 43 of 60 held-out runs with nothing unsafe, against 37 for always fast, which took 14 unsafe actions, and 35 for always deliberate. It cost almost exactly what the fast path did.

The live search overfit: it improved the training score but accepted only 38 held-out runs. That is why listing 6.4 now shows the reviewer the held-out evidence before promoting a composite. On this pass the reviewer would have kept the seed.

## Supporting material

- [The assistant's world, tools, stand-in, and evaluation code](assistant/README.md).
- [The 61 requests with their hidden truth](assistant/requests.md).
