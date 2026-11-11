# Chapter 7: Improving the agent harness by searching for code

Chapter 7 extends artifact search to code. Chapters 2 to 4 searched a prompt, chapter 5 learned values for memory entries, and chapter 6 searched a function in procedural memory. Here the artifact is a learning component: code with one entry point that stores memories and says what serves each one.

A memory's route names what serves a request that retrieves it: a decision program with no language model, the agent at a reasoning level, or the agent with no way to act. That fixed shape keeps the search inside a small area that can be measured. The search runs offline, after a day of requests, and the next day the assistant has abilities it lacked.

## Code listings

Use the project environment described in the [repository setup](../../README.md#setup), then run the commands below from the repository root. All five listings replay saved replies by default and need no API key. Run chapter 6's listings 6.2 and 6.3 first, because the assistant starts with the function and the reasoning floor they saved.

```powershell
python chapters/ch07/01_reference_component.py
python chapters/ch07/02_service_gap.py
python chapters/ch07/03_code_search.py
python chapters/ch07/04_online_week.py
python chapters/ch07/05_generalization.py
```

| Listing | Source | What to inspect |
|---|---|---|
| 7.1 | [01_reference_component.py](01_reference_component.py) | A learning component written by hand, its three memories and routes, and one request traced from words to booking |
| 7.2 | [02_service_gap.py](02_service_gap.py) | The service gap on held-out requests for the assistant at its floor, at high reasoning, and with the hand-written component |
| 7.3 | [03_code_search.py](03_code_search.py) | A search that starts from a component using only the language model, with each rewrite's score on its own requests and at the hidden gate |
| 7.4 | [04_online_week.py](04_online_week.py) | Four days of requests and three nights of improvement, with Dana's note changing on Wednesday |
| 7.5 | [05_generalization.py](05_generalization.py) | The found component, unchanged and with no search, on a day that adds chapter 6's second class of request |

Add `--record` to make paid calls for replies the recordings lack, up to `--budget` dollars of new calls. The agent is gpt-5.4-mini, the decision model is TypeSafe's Jev, and the builder that writes code is the strongest model configured in `helix.metacognition.model_settings`. Recording needs keys for all three in `.env`.

## What the recorded runs show

All five listings are recorded and replay with no key, each in under a minute. The first table measures the same 52 held-out requests five ways: 24 ask for one meeting, 8 ask for two, and 20 are near misses. The gap is the mean distance from good, where a request served right with no language model counts 0, one the language model got right counts 0.4, one not done counts 1, and a wrong action counts 3.

| Assistant | Right | Gap | Cost per request | Seconds per request | Single meetings served by |
|---|---:|---:|---:|---:|---|
| Agent at chapter 6's floor, no reasoning | 48 | 0.45 | $0.00227 | 3.02 | The agent, all 24 |
| Agent at high reasoning | 50 | 0.42 | $0.00495 | 5.64 | The agent, all 24 |
| Hand-written component, listing 7.2 | 50 | 0.09 | $0.00106 | 1.52 | A decision program, all 24 |
| Searched component, DGM, listing 7.3 | 52 | 0.12 | $0.00143 | 1.90 | A decision program for 16, the agent at low reasoning for 8 |
| Searched component, GEPA, listing 7.3 | 52 | 0.08 | $0.00123 | 1.78 | A decision program for 22, the agent at low reasoning for 2 |

The agent alone acted on 4 of the 20 near misses at its floor and on 2 at high reasoning, and every component sent all 20 on with no way to act. The hand-written component's two errors were two-meeting requests that its router read as not asking for a booking. A searched program sends a request to the agent at low reasoning when the request carries a condition the program cannot check, such as a day to avoid.

### The search, with two methods

Both searches start from a component that keeps every request on the language model, which got 17 of its own 30 requests right. A rewrite reaches the hidden gate only by clearing the bar on its own requests and serving some of them with no language model. No six-word run from a gate or held-out request appears in any component, stored program, or reply from the builder.

| Candidate | Right on its own requests | Right at the gate | Wrong actions | Outcome |
|---|---:|---:|---:|---|
| Seed, every request on the agent | 17 of 30 | not reached | 0 | Refused |
| DGM, first rewrite | 30 of 30 | 26 of 26 | 0 | Admitted and installed |
| GEPA, rewrite 1 | 30 of 30 | not reached | 0 | Refused: it served nothing with no language model |
| GEPA, rewrite 2 | 30 of 30 | 26 of 26 | 0 | Admitted and installed |

```powershell
python chapters/ch07/03_code_search.py --method gepa --population 2
```

The DGM search stopped at its first admitted rewrite and cost $1.40 in four calls to the builder. GEPA's first generation measured two rewrites and cost $2.39 in eight calls. A second GEPA generation, which is where two rewrites are crossed, is not recorded.

### A week of days and nights

Listing 7.4 serves four days and lets the decision tier improve each night. Its Monday and first night are the same requests and the same search as listing 7.3, so they replay from the same recording. Dana's note changes on Wednesday morning.

| Day | Right | Gap | Cost per request | What happened |
|---|---:|---:|---:|---|
| Monday | 80 of 80 | 0.40 | $0.00212 | The agent serves everything, and that night the search installs a program |
| Tuesday | 22 of 22 | 0.15 | $0.00150 | The program serves 8 of 12 single meetings and sends 4 to the agent at low reasoning |
| Wednesday | 12 of 18 | 0.53 | $0.00253 | One wrong booking by the program and five by the agent, then the function is quarantined and relearned |
| Thursday | 14 of 14 | 0.11 | $0.00144 | The program and the relearned function serve as before |

On Wednesday the function was quarantined after two agent runs that used it were rejected, and one call to the builder relearned it from four runs for $0.17. While it was quarantined, the component's routes still sent requests to the agent at low or no reasoning, where chapter 6's controller alone would have raised reasoning to high, and three of the day's failures happened in that window. The program made one wrong booking, so the switch that withdraws an action after two did not fire in this recording.

### A second kind of work

Listing 7.5 adds chapter 6's second class, the same meeting at the same time on several days, and runs the DGM component again with no search. The gate refused what it wrote. Its routes got all 42 gate requests right, but the single-meeting program it wrote this time handed its practice requests to the agent, so the component kept that work on the agent and nothing left the language model.

The assistant therefore stayed as it was. On the 84 held-out requests of both kinds the agent at its floor got 77 right: 24 of 24 single meetings, 8 of 8 two-meeting requests, 29 of 32 same-time requests, and 16 of 20 near misses, at $0.00364 a request. A component admitted once does not have to succeed when it runs again, because the builder writes a new program each time.

### What these runs cost, and their limits

Against the agent at its floor, the DGM component saves $0.00084 and 1.1 seconds a request and gets four more requests right, so its search is repaid after about 1,700 requests. The saving in dollars is small because chapter 6 had already brought the agent to a cheap floor. These are two searches and one week on one simulated assistant, each recorded once, and a near miss counts as right when nothing is booked, so those replies are not graded.

## The assistant and its requests

The assistant, its calendars, its checker, and its scheduling function are chapter 6's. [assistant/desk.py](assistant/desk.py) names those parts for the packaged harness in `helix.metacognition`, which holds the recorder, detector, learner, gate, and controller that listing 6.5 wired by hand. [assistant/harness.py](assistant/harness.py) holds the chapter's settings and adds no mechanism.

Chapter 6's requests come from one sentence template, so a program could pass by matching that sentence. [assistant/requests.py](assistant/requests.py) rewords each request in four styles and adds near misses, which use the words of the work but ask for something else. A separate writer produced the wordings in [assistant/banks.json](assistant/banks.json) from a brief that described only the assistant's job.

| Set | Requests | Who may see it |
|---|---:|---|
| Experience | 80 | The assistant serves these, and a component learns from the runs |
| Search | 30 | A search measures each rewrite here and shows the builder what happened |
| Gate | 26 | Only the gate, which gives the builder counts and no text |
| Held out | 52 | Nothing, until the final measurement |

## Learning mechanisms as source

Chapter 6's learning mechanisms are also held as source in [mechanisms/](mechanisms/), with one entry point each, `async def learn(lab)`. `tests/test_ch07_seed_mechanisms.py` checks that they learn exactly what chapter 6's classes learned. The sweep that first used them is kept in [chapter_appendices/ch07_learning_regime_legacy/](../../chapter_appendices/ch07_learning_regime_legacy/).
