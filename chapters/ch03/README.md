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
This chapter introduces the family, names the primitive underneath it, and
shows why no single signal is ever quite enough.

Four runnable scripts:

| § | Concept | Script |
|---|---------|--------|
| 3.1 | The proof gate, and why we measure instead | `01_godel_gate.py` (no LLM) |
| 3.2 | The Gap Function and the family of signals | `02_revenue_check.py` |
| 3.3 | The signal ceiling: two signals on one run | `03_two_signals.py` |
| 3.4 | SPO and the plateau | `04_revenue_spo_loop.py` |

The search methods that climb past where SPO stops, GEPA and DGM, are
chapter 4. This chapter ends on the wall they exist to cross.

## Prerequisites

You should have completed chapter 2. The revenue task runs entirely in
memory (`agents/revenue.py`), so it needs no corpus or index; you only need
a working LLM provider key in `.env`. The one no-LLM demo (`01_godel_gate.py`)
runs with no key and no cost at all.

---

## §3.1 The dream of a self-improving machine

A proof is the strongest signal you can have, because if you can prove a
change helps, you never have to test it. Remember, though, that we are
improving a language model wired to eight tools, and nobody can prove a
useful thing about that. So we give up the proof and keep the spirit: a
change is worth making only if we can show it helped, and the showing is a
measurement we can trust.

Before we walk away from the proof, it helps to see it once. Run the no-LLM
demo:

```
python chapters/ch03/01_godel_gate.py
```

A tiny `provably_better` gate accepts a candidate only when it can check the
whole input space and prove the candidate is correct everywhere and cheaper.
You can prove that about a sort over twenty-four inputs because the space is
tiny; you cannot prove it about a tool-wired language model, which is exactly
why the agent gets a measurement instead of a proof. Next to the proof gate
sits the `empirically_better` gate we actually use, which judges on a single
sample and can be fooled by an easy one.

---

## §3.2 The Gap Function and the family of signals

Every signal you will ever build does the same job the chapter-2 judge did,
just less noisily: it looks at what the agent produced and reports the gap
between that and good. Because the job is always the same, the framework
gives every signal the same return shape, a `GapMeasurement`, and that one
shape is the spine the whole book hangs on. A ground-truth check fills its
`score`, the chapter-2 judge fills its `preference`, and a reflection fills
its `feedback`, but it is always the same object, so the rest of the loop
never has to care which signal ran.

Think of the family as a spectrum. At one end sits proof, certain but almost
never available; at the other sits the judge, available everywhere but only
an opinion; ground truth and reflection live in between. As a general rule,
you reach for the most trustworthy signal your task can actually give you,
and most tasks cannot give you proof.

### Ground truth: precise, and only as good as your checker

On the revenue task we already know the right leaderboard, so instead of
asking an opinion we can check the agent's answer against the truth. That is
a ground-truth signal: it runs the agent and scores the result against a
known-correct value, so the same output always earns the same score. That is
as close to proof as most tasks get.

The catch is that someone has to build that checker, and a checker is itself
code that can be wrong. A confidently wrong oracle is worse than an honest
judge, because it scores a bad answer 1.0 and sends your search marching in
the wrong direction. Remember, ground truth is the most precise signal you
can have and the most expensive to get right, so you check the checker first.

```
python chapters/ch03/02_revenue_check.py
```

It runs a deliberately weak genesis prompt and a deliberately perfect oracle
twice each at temperature 0. This is the check-the-checker step: if your
oracle does not score 1.0 on a known-perfect answer, your ground truth is
broken before any search begins. The revenue check scores with plain Python
in `agents/revenue.py`; `helix.signals.task_success.TaskSuccessSignal` is the
same idea in its general, reusable form.

### Reflection: a critique with no score

A reflection signal does something a score never can: it reads the agent's
trajectory and says what went wrong in words. It fills the `GapMeasurement`
`feedback` field and leaves `score` empty, because a critique is not a
number; it is the difference between "the totals are wrong" and "you never
filtered the internal channel." This is the second new signal of the chapter,
and `helix.signals.reflection.Reflection` is the framework version that
chapter 4's GEPA improves against.

---

## §3.3 The signal ceiling: why one signal is never enough

Here is the uncomfortable truth the family hides: no single signal is
complete. A ground-truth score tells you how wrong you are but never why, and
a reflection tells you why but never how wrong. This is the
generation-verification gap, and it is the real reason a search starved of
information stalls, not just a quirk of one method.

The way to feel it is to put the two signals on the same run and watch each
fall short of the other:

```
python chapters/ch03/03_two_signals.py
```

The script runs the revenue agent once on the genesis prompt, then measures
that one trajectory two ways. Ground truth returns a `GapMeasurement` with
`score` 0.63 and no diagnosis, the symptom; reflection returns one with
`feedback` naming the three rules the agent skipped (status, channel,
payment) and no score, the diagnosis. The score has a number and no why, the
critique has a why and no number, and neither alone is enough to drive a
confident fix.

That gap is the whole motivation for the rest of the book. You escape it
either with a smarter search that squeezes more from the signal you have, or
with a richer signal that says more, and the next chapter does both at once.

---

## §3.4 SPO and the plateau

Now we put it together and run SPO on the revenue task with the Archive and
the propose-measure-select-record loop you already built in chapter 2.
Nothing about the loop is new, which is the point; what is new is the task,
hard enough to push SPO to its limit, and the ground-truth scorer driving it.
Watching a method fail well is how you learn why the next one exists.

```
python chapters/ch03/04_revenue_spo_loop.py
```

The script re-implements the chapter-2 loop inline for readability: build the
agent on the genesis prompt, hand it an SPO search and the ground-truth
scorer, and drive ten rounds. Each round rewrites the current best prompt,
scores it, and records the result to the archive with its parent pointer. The
only real change from chapter 2 is the signal dial, which has turned from a
judge's preference to a checkable score.

Here is the wall. The genesis prompt already scores 0.63, and across ten
rounds SPO cannot push past it; it generates plausible rewrites, but its only
feedback is the outcome score, which says the totals are wrong without ever
saying which of six rules it forgot. The reflection from section 3.3 knew the
answer (status, channel, payment), but SPO never sees a reflection, so it is
stuck.

The plateau has two causes, and that is the lesson. SPO is a blind search
trapped in a local optimum, and it is driven by a signal too thin to localize
the fix. Chapter 4 turns both dials at once: GEPA is a smarter search, and it
reads the reflection signal you just met, which is how it climbs to 0.86;
DGM goes further to 1.00.

| score | method | where it stops |
|-------|--------|----------------|
| 0.63 | SPO | this chapter: a blind search on a signal that cannot say why |
| 0.86 | GEPA | chapter 4 |
| 1.00 | DGM | chapter 4 |

---

## A note on cost

This chapter is the cheap one. `01_godel_gate.py` is free; `02_revenue_check.py`
and `03_two_signals.py` are a handful of agent and reflection calls;
`04_revenue_spo_loop.py` is one agent run plus one proposer call per round,
around ten rounds. The expensive search, the methods that keep many
candidates alive at once, lands in chapter 4, where the full cost ladder is
run and written up in
[`../ch04/REVENUE_SEARCH_LADDER.md`](../ch04/REVENUE_SEARCH_LADDER.md).

The honest framing carries across both chapters: the foundations come from
papers that cost research budgets to run, and the recipes here cost a few
dollars. The pattern generalizes even when the benchmark numbers do not.

---

## What's next

Chapter 4 picks up exactly where this one stops, at SPO's plateau, and
introduces the methods that climb past it. GEPA reaches 0.86 by reading the
reflection signal from section 3.3, and DGM reaches 1.00 with an archive that
never forgets. It also breaks out the lineage dashboard so you can see the
whole search as a tree, and the Godel machine returns there when DGM relaxes
its proof gate to evidence.
