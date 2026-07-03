# Chapter 3: Measuring what good looks like

In 2003 Jurgen Schmidhuber described a machine that could rewrite any part
of its own code, with one strict condition: it was only allowed to make a
change once it had proven the change would help. He called it the Godel
machine, and it is still the cleanest definition of self-improvement we
have: improve yourself, but only for the better, and know it for certain.
Real agents are far too messy to prove anything about, so this chapter is
about the practical version of that dream, measuring whether a change helped
instead of proving it.

You already built one measurement in chapter 2: an LLM-as-judge that
preferred one prompt's answer to another. That is one member of a whole
family of signals, and a judge's opinion sits at the noisy, cheap end of it.
This chapter introduces the family, runs every member against one revenue
task that carries through chapter 4, and shows why a single climber on any
one signal eventually stalls.

Six runnable scripts across four sections:

| § | Concept | Script |
|---|---------|--------|
| 3.1 | The proof gate, and why we measure instead | `01_godel_gate.py` |
| 3.2 | The revenue task, and checking the checker | `02_revenue_check.py` |
| 3.3 | Reflection and the signal ceiling | `03_two_signals.py` |
| 3.4 | Two climbers into the plateau | `04_revenue_hillclimb_loop.py`, `05_revenue_spo_loop.py`, `06_revenue_climbers_compare.py` (no LLM) |

The methods that climb past the plateau, GEPA and DGM, are chapter 4. This
chapter ends on the wall they exist to cross.

## Prerequisites

You should have completed chapter 2. The revenue task runs entirely in
memory (`agents/revenue.py`), so it needs no corpus or index; you only need
a working LLM provider key in `.env`. One script runs with no key and no cost
at all: `06_revenue_climbers_compare.py`.

---

## §3.1 Measuring the gap

You have a rewritten prompt in hand, and before you ship it you need to know
whether it actually helped. A proof would be the strongest answer there is,
because if you can prove a change helps you never have to test it. We are
improving a language model wired to eight tools, though, and nobody can prove a
useful thing about that, so we give up the proof and keep the spirit: a change
is worth making only if we can show it helped.

You can see the trade on a live agent:

```
python chapters/ch03/01_godel_gate.py
```

The chapter-2 agent (one genesis prompt artifact, no tools) answers three
questions about the line "strawberry fields are forever and ever". Arithmetic
and a letter count have a computable truth, so the `proof_gate` derives the
answer itself and its verdict is certain: the agent says 8 r's, `LINE.count("r")`
computes 7, rejected. The third question, a five-word summary, has no checker
at all, so verification degrades to measurement. That gap, proof where truth is
computable versus measurement everywhere else, is the trade every search in
this book makes.

Every signal returns the same shape, a `GapMeasurement`, and that one object is
the spine the whole book hangs on. A ground-truth check fills its `score`, the
chapter-2 judge fills its `preference`, and a reflection fills its `feedback`,
but it is always the same object, so the rest of the loop never cares which
signal ran (`helix/signal.py`).

---

## §3.2 The revenue task

Every number in this chapter and the next comes from one running example: a
small company database and one vague analytics question, "who were our top
three reps last quarter and how much did they bring in?" The agent has five
tables and eight query tools, and six rules the question never spells out (net
of refunds, the right quarter, four kinds of orders to exclude) separate a
correct answer from a plausible one. A canonical pipeline in plain Python
knows the true leaderboard, so any answer can be scored against it.

### Ground truth: precise, and only as good as your checker

Because we know the right leaderboard, we check the answer against the truth
instead of asking an opinion. The catch is that a checker is itself code that
can be wrong, and a confidently wrong oracle is worse than an honest judge,
because it scores a bad answer 1.0 and sends your search the wrong way.

```
python chapters/ch03/02_revenue_check.py
```

It runs a weak genesis prompt and a perfect oracle twice each at temperature 0.
This is the check-the-checker step: if your oracle does not score 1.0 on a
known-perfect answer, your ground truth is broken before any search begins. The
check scores with plain Python in `agents/revenue.py`;
`helix.signals.task_success.TaskSuccessSignal` is the same idea in general form.

---

## §3.3 The family of signals

Think of the family as a spectrum. At one end sits proof, certain but almost
never available; at the other sits the judge, available everywhere but only an
opinion; ground truth and reflection live in between. As a general rule, you
reach for the most trustworthy signal your task can actually give you.

### Reflection: a critique with no score

A reflection signal reads the agent's trajectory and says what went wrong in
words. It fills the `feedback` field and leaves `score` empty, because a
critique is not a number; it is the difference between "the totals are wrong"
and "you never filtered the internal channel." It is also the signal chapter
4's GEPA improves against (`helix/signals/reflection.py`).

### The signal ceiling

Put both signals on the same run and each falls short of the other:

```
python chapters/ch03/03_two_signals.py
```

The script runs the revenue agent once on the genesis prompt and measures that
trajectory two ways. Ground truth returns `score` 0.63 and no diagnosis, the
symptom; reflection names the three rules the agent skipped (status, channel,
payment) and returns no score, the diagnosis. No single signal is complete, and
that incompleteness is the wall the rest of the book climbs.

---

## §3.4 Climbing into the plateau

A measurement is only useful if a search consumes it. This section builds the
two simplest searches, both single-candidate, each fitting a different signal,
and runs both on the revenue task.

Both run inside the same loop you built in chapter 2: measure, propose,
select, record, repeat. The chapter-3 scripts re-implement that loop inline
for readability, and the framework `OfflineImprover` from chapter 2 is the
durable version with the deploy gate. Swapping the search is the only change
between them.

### Hill-climbing: the empirical Godel ratchet

The simplest search that consumes a measurement is Karpathy autoresearch:
rewrite the prompt, keep the change only if the measured score went up, repeat.
It is the Godel gate from 3.1 turned into a loop, with proof relaxed to a
number, and it is the single-thread ratchet behind Tobi Lutke's reported 19
percent overnight run. Hill-climbing reads an absolute score, which is exactly
what the ground-truth signal gives it.

```
python chapters/ch03/04_revenue_hillclimb_loop.py
```

The genesis prompt scores 0.63, and across ten rounds no rewrite ever beats it,
so the net gain is zero. A ratchet keeps a change only when the score rises, so
it can never hold a worse intermediate step, and pinning all six rules at once
needs exactly that. A ratchet can climb a hill, but it cannot cross a valley.

### SPO: climbing without a ground truth

Not every task has a checkable answer, and that is where SPO earns its keep. It
accepts a rewrite when a pairwise judge prefers it, not when an absolute score
rises, so it needs no ground truth at all; it is the method for when you cannot
check the answer directly. You met it in chapter 2; here it runs on the same
revenue task for contrast.

```
python chapters/ch03/05_revenue_spo_loop.py
```

SPO consumes a preference where hill-climbing consumes a score, but it is still
a single climber, and it stalls at 0.63 too. The signal you have decides which
search you can run; it does not decide whether a single climber gets stuck.

### Two climbers, one wall

Run the two loops, then one no-LLM script reads both run archives and prints a
signal-agnostic side by side, so the wall is reproducible rather than asserted:

```
python chapters/ch03/06_revenue_climbers_compare.py
```

```
method      genesis   best    rounds  net gain
hill-climb  0.63      0.63    10      +0.00
SPO         0.63      0.63    10      +0.00
```

Both stall at 0.63, the genesis score neither can beat. Remember, each is a
single climber trapped in a local optimum, and each is driven by a signal too
thin to say which rule is missing.

Chapter 4 turns both dials at once, a search that keeps many candidates and the
reflection signal you met in 3.3, climbing from 0.63 to 0.86 and then to 1.00.
That last step completes the Godel arc from 3.1: DGM swaps the Godel machine's
proof for measured evidence, the same relaxation hill-climbing made here, now
with an archive that never forgets.

| score | method | where it stops |
|-------|--------|----------------|
| 0.63 | hill-climb / SPO | this chapter: a single climber on a signal that cannot say why |
| 0.86 | GEPA | chapter 4 |
| 1.00 | DGM | chapter 4 |

---

## A note on cost

This chapter is the cheap one. `06_revenue_climbers_compare.py` is free (no
LLM), `01_godel_gate.py` is three one-line model calls, and `02_revenue_check.py`
and `03_two_signals.py` are a handful of agent and reflection calls. The two
loops are one agent run plus one proposer call per round, around ten rounds each.

The expensive search, the methods that keep many candidates alive, lands in
chapter 4, where the full cost ladder is run and written up in
[`../ch04/REVENUE_SEARCH_LADDER.md`](../ch04/REVENUE_SEARCH_LADDER.md).

The honest framing carries across both chapters: the foundations come from
papers that cost research budgets to run, and the recipes here cost a few
dollars. The pattern generalizes even when the benchmark numbers do not.

---

## What's next

Chapter 4 picks up exactly where this one stops, at the 0.63 plateau, and
introduces the methods that climb past it. GEPA reaches 0.86 by reading the
reflection signal from section 3.3, and DGM reaches 1.00 with an archive that
never forgets. It also breaks out the lineage dashboard so you can see the whole
search as a tree, and the Godel machine returns there when DGM relaxes its proof
gate to evidence.
