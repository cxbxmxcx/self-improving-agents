# The live pass of 2026-10-12

These files record one live pass of listings 6.4 and 6.3. Both use gpt-5.4-mini for the assistant, without reasoning on the fast path and at high effort on the deliberate path, and Jev as the decision model. The pass cost about $2.60: $0.87 for the search and $1.73 for the comparison, of which the decision models took about three cents.

| File | Contents |
|------|----------|
| `search_2026-10-12.json` | Listing 6.4's report: paired runs, threshold search, GEPA, and the online phase |
| `searched_2026-10-12.json` | The configuration and thresholds that search kept |
| `proposals_2026-10-12.jsonl` | The proposer's replies during GEPA |
| `evals_2026-10-12.json` | Listing 6.3's report: five conditions, 20 held-out requests, three trials each |

## The comparison

Metacognition with the seed configuration accepted the most requests and took no unsafe action, at almost the fast path's cost. Always fast took 14 unsafe actions in 60 runs, all of which the monitor would have held.

```text
condition                accepted unsafe  score  $/request s/request decisions regret
always fast                 37/60     14  0.479    0.00370      4.32       0.0  0.332
always deliberate           35/60      1  0.496    0.00552      5.54       0.0  0.333
seed configuration          43/60      0  0.664    0.00391      4.98       1.9  0.222
searched configuration      38/60      0  0.579    0.00518      5.34       1.9  0.255
LLM decision model          39/60      0  0.596    0.00441      4.61       1.7  0.209
```

## What the pass taught

**The search overfit the training split.** It raised the training score from 0.661 to 0.690 by learning something true of this model, that moves need deliberation, because the fast path picked the wrong event to move. On held-out requests it accepted 38 against the seed's 43, because it also deliberated on moves the user had already confirmed and stopped asking on one ambiguous request. Listing 6.4 now scores the seed and the searched configuration on the held-out requests before the reviewer decides, and on this evidence the reviewer would have kept the seed.

**Deliberation rarely paid with this model.** On the training requests built for it, the fast path accepted 8 of 9 and the deliberate path 7 of 9, at about twice the cost. `book_time`'s description already tells every path to check attendees' calendars, which is the planner's main step.

**Asking and confirming paid.** On ambiguous requests the ask path accepted every one, and the fast path guessed and acted unsafely four times in five. The monitor turned every unconfirmed send and every move of someone else's event into a question.

## Corrections made after this pass

Two held-out requests were reworded and one requirement relaxed after diagnosis, so a new pass will differ on them:

- **D12.** "Find two blocks" became "Find and book two blocks", because the model reported valid blocks without booking them.
- **D14.** "Leo" became "Leo Chen", because the deliberate path asked which Leo was meant.
- **D13.** The answer may now word the launch in several ways, not only "launch is".
