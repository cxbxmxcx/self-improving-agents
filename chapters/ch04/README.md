# Chapter 4: Evolving the agent

Life solved a search problem far harder than ours long before we existed.
With no designer and no plan, evolution found eyes, wings, and brains by
doing three dumb things over and over: vary, select, and remember. This
chapter borrows that recipe to climb out of the rut where chapter 3 left us,
because when a single hill-climber gets stuck, a population that varies and
remembers can find the path it missed.

Recall where SPO stopped, around 0.63 on the revenue task, unable to tell
which rule it had forgotten. The two methods in this chapter, GEPA and DGM,
are both evolutionary answers to that exact wall, and they climb it the rest
of the way to 0.86 and then 1.00. They cost more than SPO, so the real lesson
is not that bigger search wins, but when the extra spend is worth it.

The runnable scripts, in two groups. First the no-LLM demos that teach each
idea for free, then the LLM loops that complete the cost ladder:

| § | Concept | Script |
|---|---------|--------|
| 4.1 | Natural selection in twelve lines | `01_natural_selection.py` (no LLM) |
| 4.2 | The Pareto front | `02_pareto_demo.py` (no LLM) |
| 4.2 | GEPA reaches 0.86 by reading the trace | `03_revenue_gepa_loop.py` |
| 4.3 | Quality-diversity sampling | `04_qd_sampler.py` (no LLM) |
| 4.3 | A deceptive landscape: greedy sticks, an archive escapes | `05_deceptive_optimum.py` (no LLM) |
| 4.3 | DGM reaches 1.00 with an archive that never forgets | `06_revenue_dgm_loop.py` |
| 4.4 | Seeing the evolution: the lineage dashboard | `helix/dashboard/app.py` |

The task and scorer are the same `agents/revenue.py` from chapter 3, so the
only thing that changes between SPO, GEPA, and DGM is the search. The full
findings, including why the configuration matters, are in
[`REVENUE_SEARCH_LADDER.md`](REVENUE_SEARCH_LADDER.md).

## Prerequisites

You should have read chapter 3 and watched SPO plateau at 0.63; this chapter
is the other half of that story. The four no-LLM demos run with no key and no
cost. The two LLM loops use the same provider key and the same in-memory
revenue task as chapter 3, so there is no corpus to build.

---

## §4.1 Where hill-climbing gets stuck

Before the named methods, it helps to name the enemy they fight, the local
optimum, and the idea they all borrow, evolution. A local optimum is the
first peak a climber reaches, good enough to trap it and not the highest peak
around. Evolution beats it not by climbing smarter but by keeping many
climbers and a long memory, so the population can wander off the first peak
and stumble onto a better one.

Strip evolution to its mechanics and there are four: variation, heredity,
selection, and time. Map them onto our framework and the biology turns into
code you already have, variation is `mutate()`, heredity is the parent
pointer every artifact carries, selection is the Signal, and time is one
search round after another. The cleanest way to feel this is a tiny loop with
no model in it at all:

```
python chapters/ch04/01_natural_selection.py
```

It evolves integers toward a hidden target, climbing purely on variation and
selection, and reaches the target in a dozen generations. There is no
cleverness in it, only the three dumb moves repeated, which is the whole
point.

---

## §4.2 GEPA: reflection, population, and the Pareto front

GEPA, short for Genetic-Pareto, is evolution with two ideas SPO did not have.
It keeps a population instead of one line, and it lets a variant read the
agent's execution trace before it mutates, so the change can target the real
failure instead of guessing. The third idea, a Pareto front, lets a
high-quality answer and a cheaper one survive together, and the three carry
the revenue task from 0.63 to 0.86.

The biggest single reason GEPA beats SPO has nothing to do with populations:
it reads how the agent failed, not just that it failed. Where SPO's proposer
hears only that the totals are too high and must guess the missing rule,
GEPA's reflector sees the actual tool calls, notices the agent filtered status
but never channel or payment, and writes that exact fix into the next prompt.
That is the difference between a blindfolded edit and a targeted one, and it
is most of the climb to 0.86.

Run the GEPA rung:

```
python chapters/ch04/03_revenue_gepa_loop.py
```

### The Pareto front

A population only helps if you keep the right members, and right usually means
good at different things. The Pareto front is the honest way to keep them: a
candidate survives if nothing beats it on every objective at once, so the
survivors are the best-at-something trade-offs rather than one winner. See it
for free first:

```
python chapters/ch04/02_pareto_demo.py
```

A short honesty note. In the revenue loop the signal returns a single score,
so its front collapses to a ranked list; the genuine two-objective front lives
in `02_pareto_demo.py`. And the framework class `helix/search/gepa.py` constructs
a reflector but never calls it, feeding the judge's feedback to mutation
instead, so the loop that actually reflects on a trajectory is the one you run
here, `03_revenue_gepa_loop.py`.

---

## §4.3 DGM: the archive that never forgets

GEPA's weakness is the weakness of any small population: it converges. After a
few generations every candidate is a minor variant of the same idea, the
diversity is gone, and the odd branch that would have cracked the last rule
died generations ago. DGM, the Darwin-Godel Machine, fixes this with one
stubborn rule, never throw anything away, and that is what carries the task
the final step to 1.00.

Remember the Godel machine from chapter 3, the one that would only change
itself with a proof in hand. DGM is where that idea comes back, with the proof
deliberately relaxed to evidence: it self-modifies whenever the archive shows
a change worked, not when it can prove it will. That is the honest reading of
the name, Darwin for the evolving archive and Godel for the self-modification,
with the proof swapped for evidence from the archive.

### Quality-diversity and the recovery to 1.00

An archive that never deletes is only useful if you go back into it, and
quality-diversity sampling is what sends you back. Instead of always breeding
from the current best, you weight every candidate by how good it is and how
rarely you have used it, so a strong-but-neglected branch keeps getting a
turn. Two no-LLM demos make it concrete:

```
python chapters/ch04/04_qd_sampler.py
python chapters/ch04/05_deceptive_optimum.py
```

The first shows greedy fixating on the top score while quality-diversity keeps
every branch alive. The second is the no-cost mirror of this whole chapter: a
greedy climber sticks at a tempting local peak while an archive search crosses
the valley to the taller one, 0.63 versus 1.00 with no model in the loop.

Run the DGM rung:

```
python chapters/ch04/06_revenue_dgm_loop.py
```

On the revenue task DGM revisits a low-scoring branch GEPA had thrown away,
finds the one phrasing that excludes both cancelled and returned orders, and
because nothing is discarded, it keeps the 1.00. One honesty note: the
framework's quality-diversity sampler is a content-hash stand-in, so the real
diversity weighting is the `score / 2 ** times_sampled` decay you see in
`04_qd_sampler.py` and the loop's own `sample_parent`.

---

## §4.4 Seeing the evolution: the lineage dashboard

We have talked about populations and archives as if you could see them, so
let's finally see them. Every prompt we evolved is an immutable artifact with
a pointer to its parent, which means the whole run has been a tree all along,
we just never drew it. In this last section we break out the Helix dashboard
and look at the evolutionary hierarchy directly.

You do not need a special structure to draw the tree, because immutable
versioning already built it. Every `mutate()` stamps the child with its
parent's id and version, a genesis artifact has none, so the edges are simply
child-points-to-parent and one archive call, `lineage_tree()`, walks them into
a forest. The GEPA and DGM loops record each candidate with its score and
method, so the tree comes up colored and ranked.

After running the two loops, launch the viewer:

```
streamlit run helix/dashboard/app.py
```

Point it at `chapters/ch04/runs/revenue_gepa.sqlite` or
`revenue_dgm.sqlite`, focus the prompt artifact, and open the Lineage panel.
GEPA's tree is a shallow fan that narrows as the population converges, while
DGM's is a deep, branching archive with one long shoot reaching down to an old
node and back up to 1.00, the recovery you read about made literal. Remember
that the gold crown marks the highest-scoring artifact, not the one in
production; promoting a champion to live is the same deliberate gate you built
in chapter 2.

---

## What's deferred

- **Memory-grounded search**: evolutionary search where the artifact is a
  memory entry, not a prompt. Chapter 5.
- **Online evolutionary search**: continuous mutation against live traffic,
  no labeled eval set. Chapter 5, with HITL and feedback.
- **Code-level DGM**: aiming the archive at a tool's implementation rather
  than a prompt, where the agent edits its own code. Later chapters, once
  sandboxed execution joins the framework.
