# The production-scale test

Written on November 10, 2026, before any of its runs. Nothing in this plan has been run on a paid model. The pass criteria in section 7 are to be committed before the first run, as the frozen-method test's were in [DOMAINS.md](DOMAINS.md).

Two scripted tests that call no model were run the same day, and section 3 reports them: a stream of 100,000 requests served one at a time, and a load test with up to 128 requests in flight. They test the loop's own mechanics at scale. They do not test how a model behaves at that length, which is still the job of the paid runs planned below.

## 1. The claim under test

The claim is about one kind of agent: a task worker that serves thousands of requests, acts through tools, and gets a verdict on what it did. For that agent, the metacognitive harness with offline search should make serving cheaper and faster, keep it at least as accurate, make it more predictable, and let it recover by itself when conditions change. The claim also says the result repeats, across independent searches and across desks, including one we did not write.

| Part of the claim | What it means in this test |
|---|---|
| Cheaper | Lower total cost per request, with every learning and search call counted |
| Faster | Lower response time on average, with a slow tail that is no worse |
| As accurate | At least the same share of requests done right, on the same requests |
| Safer | Fewer wrong actions, and almost none by a decision program |
| More predictable | The same request gets the same result when it is served again |
| Robust | After each change of conditions the system returns to its earlier level with no manual step |
| Repeatable | Most independent searches are admitted, on every desk |

Three things stay outside the claim until they are tested on their own. A request that needs several actions still goes to the agent, because a decision program takes one action. An agent with no verdict on its outcomes cannot run the loop as built, and open-ended goal solving has so far shown lower reasoning at best, never a program.

## 2. Where the evidence stands

Everything so far is simulated. No stream served by a real model is longer than about 200 requests, and the only long streams are scripted. On six desks a search that starts from a component using only the language model produced a component the hidden gate admitted, within three rewrites, when a strong model wrote the code. The table says how far each part of the claim is supported today.

| Part | What we have | Standing |
|---|---|---|
| Cheaper and faster | Every desk with an admitted component: cost and time each fell to between about 7% and 63% of the baseline it was compared with | Shown, in simulation |
| As accurate | Held-out accuracy rose on five desks and fell on one, the on-call desk, from 50 to 47 of 52 | Mostly shown |
| Safer | Near misses stopped being acted on. A change of conditions costs two wrong program actions when requests are served one at a time, and about the requests in flight when they are served together: 102 of 400 at 128 at once, against 300 with no loop | Contained, and the bound is known |
| More predictable | Never measured | Not tested |
| Robust | With a real model, one change of conditions per run, and the calendar desk's day of the change was 12 of 18 right. Scripted, 169 changes in 100,000 requests were each repaired the same day with no stalled day, the fast path was back within 9 requests of each change, and 168 of 169 were repaired the same day at 24 requests a day | Mechanics shown. One paid event |
| Repeatable | Admitted on all six desks, and three of four first rewrites on the calendar desk. One to three runs per setup, and one component that ran again was refused | Partly |
| Thousands of requests | Scripted: 100,000 requests at about 3.2 ms of harness time each from the first block to the last, a store that ends at 104 MB, and 90 requests a second from one process at 128 in flight. Paid: nothing longer than about 200 requests, and payback of about 90 to 1,700 requests is still arithmetic | Mechanics shown. Model behavior not tested |
| A desk we did not write | None | Not tested |

Zero wrong actions in 52 requests only shows that the rate is below about 6%. A stream of thousands is what turns "none seen" into a rate that means something.

## 3. The scripted scale tests

Both tests use the calendar desk of Chapter 7 with scripted stand-ins for its three models, so a run costs nothing but time. The agent books what its function returns, and acts on one near miss in five when nothing stops it. The builder returns known code that reads only the note in force when it is asked. The decision model misreads one near miss in a hundred. The tests therefore show what the harness does with its limits, its state and its recovery. They show nothing about what a model does.

```
python -m examples.scale.long_run --requests 100000
python -m examples.scale.load
```

### 3.1 What broke, and what changed

The first long stream stalled, and each later run found something else. Every row below is a fault that a scripted run showed, with the change that answers it. A change marked as a setting is off unless a deployment turns it on, so the recorded listings of Chapters 6 and 7 replay as before.

| What a run showed | Change |
|---|---|
| After the third change of note the function could not be relearned, because its budget counts attempts per day on the wall clock. Programs served nothing for the last 220 of 600 requests | Setting: `LearningBudget(runs_per_day=...)` counts a day in recorded runs |
| After the fourth change the tier could not re-test its program, because it allows four improvement attempts in its lifetime. Bookings stayed off for 80 requests although the program was right again | Setting: `ProgramTier(long_running=True)` counts attempts and searches for each gap, and a re-test still runs when they are used up |
| A request that a program served wrongly before the function was relearned failed every later re-test, and the fast path stayed off for six days | Under `long_running`, a wrong action from before the latest relearning is no longer a gate case |
| A stretch with the calendar down was taken for a stale function, with four quarantines and three relearnings | A tool that cannot reach its service raises `Unavailable`, and that run is not held against a procedure |
| One relearning made 242 builder calls, because the learner was handed every run since the quarantine | Setting: `Metacognition(experience_limit=...)`, and the store reads only that many runs |
| The store grew by about 25 KB a run. With old episodes dropped it still grew by 2.7 KB a run, because SQLite leaves a shrunken row's page as it was | Setting: `MetaStore(keep=...)` keeps the latest episodes whole and repacks itself every 4,096 runs. An old run then holds about 0.5 KB |
| At 24 requests a day, a function that failed one use in six was never quarantined. The fast path stayed off for a week, and seven builder calls rewrote a component that had nothing wrong with it | Under `long_running`, a failed re-test or a switched-off action puts each function under the program through its own gate. One that fails is quarantined at once, and the tier waits while it is relearned |
| With many requests in flight, those already offered a function went on calling it after it was quarantined | A trusted version answers only while it is still the trusted one. A call in flight gets an error to work around |
| The harness spent about 14 ms on every request, most of it parsing and compiling a program and a function that had not changed | Each source is checked and compiled once. Every request still runs in a namespace of its own |
| With the old limits, a quarantined function's backlog was read in full every few requests, and time per request reached 329 ms by day 12 | The latest runs are read from the newest back |
| An action that was switched off stayed off until the night, so programs served 1.5% of the requests on a day of change | Setting: `ProgramTier(retest_by_day=True)` tests the action again as soon as the function under it has been relearned |

### 3.2 A stream of 100,000 requests

The stream is 400 days of 250 requests with a night after each, as listing 7.4 runs a week. Dana's note changes every second day. That is far more often than a real desk would see, and it is meant as a stress. A stretch of 60 days has no change at all, 11 days have garbled requests, and on 6 days the calendar is down for 40 requests. A second stream runs the same 400 days at 24 requests a day, where every detector has less to go on.

| Measure | 250 requests a day | 24 requests a day |
|---|---|---|
| Requests | 100,000 | 9,600 |
| Right | 99,385 (99.4%) | 9,107 (94.9%) |
| Right on a calm day | 99.9% | 99.9% |
| Changes of note | 169 | 169 |
| Repaired by that night, with the function trusted and the action on | 169 | 168 |
| Back to normal on the next calm day | 151 of 151 | 150 of 151 |
| Stalled days, and false alarms on calm days | 0 and 0 | 0 and 0 |
| Wrong answers on a day of change, mean and most | 2.3 and 5 | 2.6 and 4 |
| Wrong actions by programs on a calm day | 0 | 0 |
| Builder calls, and how many fell on a day with no change | 679 and 0 | 679 and 4 |
| Relearnings the budget skipped | 0 | 0 |
| Longest night | 0.8 s | 0.3 s |
| Store at the end | 105 MB | 59 MB |

The harness's own time on a calm day stays between 3.1 and 3.4 ms a request in every block of 5,000 requests, so the length of the stream adds a tenth at most. The whole stream runs in under eight minutes.

Three things in the table need a word. The share right falls at low volume for two reasons: each change costs the same two or three wrong answers out of 24 requests and not out of 250, and a calendar that is down for 40 requests takes half of a 24-request day. The one change at low volume that was not repaired by night was quarantined too late in the day to gather the four runs a relearning needs, and it was repaired the next morning. And the store is bounded for each run but not in total: about 50 MB for the latest 2,000 episodes, plus 0.5 KB for every run ever served.

Both streams in the table test a switched-off action again as soon as the function under it has been relearned. Chapter 7's listings wait for the night to do that, and so did the first runs of these streams. The difference is the fast path on a day of change.

| Stream | Changes that switched the action off | Back the same day | Requests until it was back, mean and most | Served by programs on days of change | Served by programs in all |
|---|---|---|---|---|---|
| 250 a day, re-test at night | 166 | 0 | none | 1.5% | 31,732 |
| 250 a day, re-test by day | 166 | 166 | 5.8 and 9 | 54.6% | 54,183 |
| 24 a day, re-test at night | 161 | 0 | none | 7.7% | 3,274 |
| 24 a day, re-test by day | 161 | 160 | 5.8 and 11 | 38.9% | 4,551 |

Everything else in the first table is the same either way, down to the count of requests right. A re-test asks no builder, so it is only a question of when.

### 3.3 Many requests at once

One process serves a batch of 400 requests with a limit on how many are in flight. A scripted agent call takes 0.5 seconds, a decision-model call takes 0.1 seconds, and a verdict arrives 0.2 seconds after its action.

| In flight | Requests a second | Mean time | 95th percentile | Right |
|---|---|---|---|---|
| 1 | 1.1 | 0.88 s | 1.86 s | 80 of 80 |
| 8 | 8.9 | 0.88 s | 1.86 s | 400 of 400 |
| 32 | 32.3 | 0.90 s | 1.87 s | 400 of 400 |
| 128 | 90.2 | 0.96 s | 2.08 s | 400 of 400 |

The mix of outcomes is the same at every level: half served by programs, a quarter handed to the agent read-only, and a quarter served by the agent. With no model time at all, one process carries 220 to 300 requests a second, and that is where a single process stops scaling. Before each source was compiled once, it stopped at 65.

The second half of the test changes Dana's note as a batch starts.

| In flight | Wrong by programs | Wrong by the agent | Right | Relearnings |
|---|---|---|---|---|
| 1 | 2 | 1 | 77 of 80 | 1 |
| 8 | 4 | 0 | 396 of 400 | 1 |
| 32 | 16 | 8 | 376 of 400 | 1 |
| 128 | 64 | 38 | 298 of 400 | 1 |
| 128, with no loop | 200 | 100 | 100 of 400 | 0 |

A harness can react only to a verdict it has. Every request already in flight when the first wrong verdicts arrive has used the stale function, so a change costs about the bookings in flight, once. The counts at 32 and 128 move by a few from run to run, since they depend on which requests finish first. After that the function is quarantined, requests in flight are stopped from calling it, one relearning starts however many requests saw the fault, and the batch ends right. With no loop the stale function serves every booking until a person notices.

### 3.4 What the stream would cost

No model was called, so cost here is arithmetic. Each request is priced at the recorded average of the arm that describes the state the system was in. Chapter 7's recordings give $0.00227 and 3.0 seconds for the agent with its learned function, $0.00106 and 1.5 seconds with the hand-written component (the one the scripted builder returns), $0.00143 and 1.9 seconds with the component the search found, and $0.00495 and 5.6 seconds for the agent alone at high reasoning. A builder call that rewrites the function is $0.165, and the search is $1.40. A request served while the program's action is off is priced as the agent with its function: the first day, and the six requests or so after each change.

| Assistant | Serving | Learning | Per 1,000 requests | Mean time |
|---|---|---|---|---|
| Agent alone at high reasoning | $495 | none | $4.95 | 5.6 s |
| Agent with Chapter 6's loop | $227 | $111 | $3.39 | 3.0 s |
| Full system, priced as the searched component | $144 | $113 | $2.57 | 1.9 s |
| Full system, priced as the hand-written component | $107 | $113 | $2.20 | 1.5 s |

Learning is about half of the full system's bill here, and that is the stress schedule at work: 169 relearnings of four builder calls each. Chapter 6's loop relearns the function too, so that cost is the same in both rows. Re-testing only at night would put the full system at $2.69 to $2.91. How often conditions change decides the rest.

| Requests between changes | Chapter 6's loop | Full system, searched component | Full system, hand-written component |
|---|---|---|---|
| 500 | $3.59 | $2.76 | $2.39 |
| 2,500 | $2.54 | $1.70 | $1.32 |
| 10,000 | $2.34 | $1.50 | $1.12 |
| 100,000 | $2.28 | $1.44 | $1.06 |

The table is in dollars per 1,000 requests, with four builder calls for each relearning, as the stream made. Listing 7.4's recorded relearning took one.

### 3.5 What these tests do not show

A scripted agent is right whenever its function is right, and a scripted builder always writes a function that passes its gate. A real agent adds mistakes of its own, and a real builder is sometimes refused.

The tests keep Chapter 7's rhythm of serving by day and improving by night. A rewrite of the component or a search still waits for the night. Only the re-test moved into the day, and it needs a gate that is already current when the function has been relearned, a few requests after the change. The stream hands it that morning's held-back requests. A deployment has only the labeled requests that arrived since the change, which may be too few to trust for a while.

The load test runs one process on one machine. Several processes on one store are not measured. Neither is the sandbox that starts a new process for every call to a learned function, which cost about 80 ms a call on this Windows machine when a served day was profiled, and would be the first limit under load.

The requests held back each night come from that day's own conditions. The stream arranges that, and a deployment has to arrange it too, with shadow requests labeled by their verified outcome. The night check of a function is only as current as its gate's cases.

## 4. Changes to the method before it is frozen

Each change answers something we measured. All but the last are made, as settings that a long deployment turns on, and each has run with scripted models. Everything under `helix/` is frozen at the commit that holds this section, and nothing there changes until every run in this plan is done.

1. **Held-back requests come from the day's traffic.** Each day a share of that day's requests is set aside and is never served by the system under test, never shown to a component, and never shown to the builder. One part is that night's gate, one part is what a search may measure on, and a few weeks judge a relearned function. In production these would be shadow requests labeled by their verified outcome, and in the simulation they are fresh requests drawn under the day's conditions. *Done in both streams.*
2. **Improvement attempts are limited for each gap, not for a lifetime.** The count of attempts and of searches starts again when a program is installed or re-tested successfully. A gap that stays open still stops costing money after its attempts are used. *Done, under `long_running`.*
3. **The learning budget counts a day in runs, not on the wall clock.** A run that compresses weeks into hours otherwise hits a daily limit that was meant for real days. *Done, with `runs_per_day`.*
4. **A stale function under a program is found before anyone rewrites the program.** When a re-test fails or an action is switched off, each function the program stands on goes through its own gate first. *Done, under `long_running`.*
5. **While a function is quarantined, the controller's raised reasoning wins over a component's route.** In listing 7.4 three of Wednesday's six failures happened because a route kept the agent at low or no reasoning during the quarantine. *Done, under `long_running`.*
6. **A switched-off action is tested again as soon as the function under it has been relearned.** Waiting for the night left the fast path off for the rest of every day on which something changed. *Done, with `retest_by_day`.*
7. **A function is put through its own gate before the detector may quarantine it, and a learner is handed only the runs that say something about its function.** Two rejected runs in four uses quarantine a function, and the runs can be rejected for the agent's own mistake. Over thousands of requests that would start paid relearnings of a function with nothing wrong with it. A near miss is right when nothing is done, and as experience it cost a builder call for a candidate written from a run that never needed the function. *Done, with `confirm_quarantine` and `learn_from_gating_runs`.*
8. **A decision model picks the function a request needs once an assistant has more than one.** Matched by their words, requests for the two kinds of booking went to the wrong function between one time in eight and one time in three, across seven writers. *Done, with `route_with`; a gate can hold cases for each function, and a running assistant can be given a new function to learn.*
9. **Optional: a decision program may take several actions.** This would move two-meeting requests off the language model and is the first step toward the goal-solving half of the claim. It is a larger change, and the test runs without it. *Left out.*

## 5. The streams

Each desk gets one stream of 4,000 requests, served as 20 days of 200 with a night after each day. Conditions change five times on the calendar desk, and each change is a kind that production agents meet. The other desks follow the same shape with their own changes.

| Day | Change on the calendar desk | What it tests |
|---|---|---|
| 5 | Dana rewords her note to ask for half an hour, and the checker enforces 30 minutes | A rule changes under a learned function |
| 8 | Two more groups of writers start sending requests | The way people word requests shifts |
| 11 | The calendar cannot be read for one request in ten, for one day | A dependency misbehaves |
| 14 | Requests for the same meeting at the same time on several days arrive, 15% of traffic | A new kind of work |
| 17 | Dana changes her note again, to three quarters of an hour, and the checker enforces 45 minutes | The same rule changes a second time |

The stream is built by `examples/scale/desk_stream.py`, and these are its particulars.

- **Mix.** Half the requests ask for one meeting, 12% for two, 30% are near misses and 8% are out of scope. From day 14 the new kind takes 15% and the rest keep their proportions.
- **Weeks.** Every request has a week of its own, drawn as Chapter 6 draws its weeks, and words no other request has, so no reply recorded for one request can answer another. The exception is out-of-scope requests: several of their templates name nobody, so about one request in forty repeats an earlier out-of-scope request word for word and gets the reply that one got. Once a note has changed, every other request for a booking sits on a week the change decides: Dana is invited, and each reading of her note that a function written before the change would make books a start the checker rejects.
- **Words.** Seven separate writers each wrote a group of 40 request templates from a description of the assistant's job and of the placeholders. None saw the method or any result. Four groups send requests from day 1, two more from day 8, and the seventh words a final exam of 200 requests that nothing sees before the last night. The templates are in `examples/scale/banks_calendar.json` exactly as returned.
- **Held back.** Each day two sets of 16 requests under that day's conditions are never served: one is the gate, the other is what a search may measure on. Twelve weeks, and six of the new kind from day 14, judge a relearned function.
- **The new kind of work.** On day 14 the team declares what the new work needs, as a second function's contract with no code. The loop has to learn that function from the agent's own accepted runs, and until it does the agent serves those requests with raised reasoning.
- **The second change of note.** It is worded in words, not digits, so that a function relearned for "half an hour" is stale again. A function that had learned to read any wording would pass it, and that would be a result too.

| Desk | Why it is in the test | Status |
|---|---|---|
| Calendar assistant | The book's example, with the most evidence so far | Built, and soaked with scripted models |
| Invoices or billing | A desk that already passed at small scale, with a different shape of work | Built; needs a generator for thousands of tasks |
| A desk we did not write | Requests written by real people, with labels we did not choose | Not built; needs two public datasets, see section 10 |

## 6. Arms and measures

Three assistants serve requests from the same stream (`examples/scale/desk_run.py`). The agent alone at high reasoning, with no learned function, is what a team deploys first, and it serves every 25th request with nothing learned, 160 in all. The agent with Chapter 6's loop serves every fifth request online, 800 in all. The full system serves all 4,000 online, with a night after each day. All three have the settings of section 4 that apply to them, so the full system differs from the loop by its decision tier and nothing else.

| Measure | How it is taken |
|---|---|
| Common requests | The requests two assistants both served: 800 with the loop, 160 with the agent alone |
| Serving cost | What a request's own model and decision-model calls cost, from each call's recorded token usage |
| Learning cost | Everything else the assistant's recordings hold: every builder call, and the agent's runs at night |
| Cost of the whole stream | An assistant that served a share of the stream is scaled to all of it: its serving cost times 5 or 25, plus its learning cost once, since learning follows changes and not volume |
| Response time | Recorded model and decision-model time; mean, median and 95th percentile |
| Share right | The desk's checker on every request, with a 95% interval |
| Wrong actions | Requests answered wrongly by booking something, counted separately for decision programs |
| Recovery | Requests from each change until the share right over 40 requests is back within 3 points of its level over the 200 requests before the change |
| Repeat consistency | After the last night the full system and the loop take the final exam three times, each from new recordings, with nothing learned, quarantined or switched off in between. The measure is the number of the 200 requests with the same verdict in all three servings, so it counts only what the models answer differently |
| Payback | The request at which the full system's cumulative cost falls below each baseline's and stays there |

Search repeats are separate from the stream. Five independent searches start from the same component on Chapter 7's own requests, each with the day of experience the chapter recorded and none of the builder's recorded replies, so every search asks the builder afresh. Each admitted component serves that chapter's 52 held-out requests.

`examples/scale/desk_report.py` computes every measure and criterion, the same way for a scripted soak and for the paid run.

The whole stream was soaked first with scripted models that misbehave: an agent that acts on one near miss in five and books one day of three of the new kind of work when it has no function for it, a decision model that misreads one near miss in a hundred and sends one request in fifty to the wrong function, and a builder that writes one function in seven for the note before the one in force. All four parts ran to the end with no fault in the driver. The soak's numbers say nothing about the claim, since scripted models take no time and cost what they are scripted to.

## 7. Pass criteria

Each criterion is measured on one desk and reported as measured. The comparison in criteria 1 to 5 is with the agent that has Chapter 6's loop, on the 800 requests both arms served, except that criterion 2 compares the cost of the whole stream and the first half of criterion 5 counts every request a program served. Criterion 9 compares the whole stream's cost, and the share right on the 160 requests all three served. None of these numbers changes after the first run.

| # | Criterion | Pass if |
|---|---|---|
| 1 | Cheaper to serve | The full system's serving cost per request is at most 70% of the baseline's |
| 2 | Pays for its learning | Its total cost, with every learning and search call, is below the baseline's by the end of the stream |
| 3 | Faster | Its mean response time is at most 70% of the baseline's, and its 95th percentile is at most 110% |
| 4 | As accurate | Its share right is no more than 2 points below the baseline's |
| 5 | Safer | Decision programs take wrong actions on at most 0.5% of the requests they serve, and total wrong actions are no more than the baseline's |
| 6 | Robust | After every change it recovers within 200 requests with no manual step |
| 7 | More predictable | Its repeat consistency is at least the baseline's |
| 8 | Repeatable | At least four of five searches are admitted within six rewrites, and each admitted one gets at least as many held-out requests right as the agent at its floor |
| 9 | Worth it against a first deployment | Its total cost is at most 25% of the agent alone at high reasoning, and its share right is at least as high |

The pattern **holds** on a desk if all nine pass. It **holds in part** if 4, 5 and 6 pass and a cost or time criterion fails, and it **does not hold** otherwise. The claim is established for agents of the kind in section 1 only if the pattern holds on every desk that is run, including the one we did not write.

Criterion 2 is the one most at risk, and that is deliberate. On the calendar desk the search saves about $0.0009 a request against Chapter 6's floor and costs $1.40 to $2.40, so each search needs about 1,700 requests to repay. A stream with several searches and relearnings may not repay them all in 4,000 requests, and the test should show that if it is so.

## 8. Rules of the test

1. One run per desk. There are no reruns because a result is poor, and a run the cap stops counts as what it reached.
2. A crash from a bug in a desk or a driver is fixed and the run starts over, and the report says so.
3. Separate writers who see neither the method nor any result write all request text.
4. A scripted soak of the whole stream, with an agent and a builder that misbehave, runs before any paid call. Each paid run is pre-flighted in replay mode and runs alone on the machine. The parts run one at a time in this order, each under its own cap and each replayed with no key before the next starts: the live check, the agent alone, the loop, the full system, and the search repeats, whose cap is what remains of the desk's $40.
5. Nobody tunes against the stream or the final exam. A defect in a world, such as an impossible task, may be fixed before the paid run and nothing else.
6. Recordings of about 30 MB per desk are kept compressed outside git, and the results and the figures' data are committed.

## 9. Cost and run time

The estimates use what the Chapter 7 recordings measured: about $0.0022 and 2.8 seconds for a request the agent serves, $0.00015 and 0.3 seconds for one a program serves, $1.40 to $2.40 for a search, and $0.17 to $1.20 for relearning a function. My estimate for the GEPA run was low by half, so these are ranges and each part has a hard cap. A recording stops before a call that could pass its part's cap, and a part the cap stops counts as what it reached.

| Part, calendar desk | Estimate | Cap |
|---|---|---|
| Full system: 4,000 requests with nights, the exam and two repeats | $10 to $17 | $18.50 |
| Agent with Chapter 6's loop: 800 requests, the exam and two repeats | $5 to $6 | $6.50 |
| Agent alone at high reasoning: 160 requests | $1 | $2.00 |
| Five search repeats | $7 to $12 | What remains of $40, about $11 |
| A live check of the wiring before the runs | under $0.50 | $0.50 |
| The decision model's calls, which the caps do not stop | about $0.60 | |
| Total | $24 to $37 | $40 |

The parts run one after another and alone on the machine: about three hours for the full system, a little over an hour for the loop, and under an hour for the rest. Building the streams, the banks and the soak costs nothing in model calls. The desk we did not write needs more building than the other two.

## 10. Decisions for the author

| # | Decision | Recommendation |
|---|---|---|
| 1 | The scope in section 1 | State it in the chapter as the conditions the pattern needs: recurring kinds of request, a verdict on outcomes, single actions, and a strong builder |
| 2 | The method changes in section 4 | Decided on November 11: changes 1 to 8 are made and tested without a model, and the method is frozen with them |
| 3 | Programs that take several actions | Leave out of this test, and decide after its results |
| 4 | Which desks, and in what order | Calendar first under a $40 cap, then review; the second desk and the outside desk after that |
| 5 | The outside desk's data | Two public sets of requests written by real people, Banking77 (about 13,000 banking queries in 77 categories, CC BY 4.0) and the out-of-scope queries of CLINC150 (CC BY 3.0). Both need a download, which needs the author's permission |
| 6 | Committing this plan | Commit the criteria before any run, so they cannot move afterward |
| 7 | Re-testing by day | Decided on November 11: yes. It is change 6 in section 4, and on the scripted stream the fast path was back within 9 requests of each change |
