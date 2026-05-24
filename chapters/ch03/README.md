# Chapter 3: Searching for better text with evolutionary methods

Chapter 2 introduced one search method (SPO) and one artifact under
improvement (the system prompt). Chapter 3 widens both axes. The reader
learns two evolutionary search methods built from scratch — GEPA and
DGM — and applies them to a *tool description* artifact, demonstrating
that the same search machinery works on any L1 text artifact, not just
the system prompt. The chapter closes by running SPO, GEPA, and DGM
together on the same tool description with the framework's
multi-improver pattern.

Four sections, five runnable scripts:

| § | Concept | Script |
|---|---------|--------|
| 3.1 | The economics of evolutionary search + tool descriptions as artifacts | `tool_description_loop.py` |
| 3.2 | GEPA from scratch: population, reflective mutation, Pareto over two objectives | `minimal_gepa.py` |
| 3.3 | DGM from scratch: archive of variants, quality-diversity sampling | `minimal_dgm.py` |
| 3.4 | Three methods on one artifact: SPO + GEPA + DGM via the multi-improver pattern | `dual_improver.py`, `escalating_improver.py` |

This chapter is denser than Ch 2 (~30 pages versus ~22). The pedagogy
remains "build it twice": each method is hand-written first in 80 lines,
then bridged to its framework class so the reader sees the algorithm
naked before they see the production wrapper.

## A note on cost

The DGM paper (Zhang et al., 2025) reports an improvement from 20% to
50% on SWE-Bench. The reported compute cost: roughly **$22,000**. The
AlphaEvolve paper (Romera-Paredes et al., 2024) is in the same range.
That's research-budget territory.

This chapter teaches the same patterns at **practical cost**: ~$2-5 per
full chapter run on Haiku + Sonnet, with the eval slice tunable down to
~$0.50 per run if you want to iterate cheaply. The trade is fidelity to
benchmark performance: we don't expect a 30-percentage-point lift on
SWE-Bench from $2 of compute. What we expect is a working demonstration
of the patterns that scaled the published results, applied to an artifact
you can actually afford to search over.

This is a central theme of the book: foundations from papers, recipes
that real engineering teams can run.

## Prerequisites

You should have completed Chapter 2 and have a working LanceDB corpus at
`data/helix_corpus.lance/`. If you haven't:

```
python ingestion/build_index.py
```

You should also have a populated `chapters/ch02/runs/helix_archive.sqlite`
from running `spo_offline_loop.py` at least once — Ch 3 inherits Ch 2's
archive and extends it.

---

## §3.1 The economics of evolutionary search

Chapter 2's SPO works one candidate at a time and accepts on win.
Evolutionary search keeps *many* candidates alive — a population (GEPA)
or an archive (DGM). The reasons to do this:

1. **Escape local optima.** SPO's hill climb can converge to a prompt
   that's locally good but globally limited. Population/archive methods
   maintain diversity, so the search can recover from a dead end by
   sampling a different branch of history.
2. **Multi-objective optimization.** When a candidate is better on one
   axis (quality) but worse on another (cost), single-objective search
   has no way to keep it. Pareto selection (GEPA) keeps the non-dominated
   front.
3. **Reuse of distant ancestors.** DGM's archive lets a mutation in
   round 20 fork from a candidate produced in round 3, not just from
   round 19's winner. Good ideas don't die when they're temporarily
   out-performed.

The cost of all of this is more evaluations per round. A population of 4
costs 4× SPO per generation. An archive of 100 needs sampling logic
but the per-round cost depends on how many you mutate per round.
Practical recipes pick small populations and short archives because
LLM-as-judge dominates the cost line.

### Tool descriptions as artifacts

A tool's *description* is the natural-language string the LLM reads to
decide whether to call the tool. It is an L1 text artifact in the same
sense the system prompt is: improvable by SPO, GEPA, or DGM without
touching code.

The framework's `TextDescriptionTool` (SPEC §16.2.2) wraps a plain
Python callable with an artifact-backed description. The implementation
stays a normal Python function. The description is a `TOOL_DESCRIPTION`
artifact in the archive. Search methods mutate the description; the
agent reads the current live description on every request.

Run the warmup:

```
python chapters/ch03/tool_description_loop.py
```

The script builds an agent whose `retrieve` tool's description lives in
the archive. An `OfflineImprover` targets the tool description and runs
three rounds with SPO. The reader sees that everything they learned in
Ch 2 §2.4 carries over: the same Improver, the same Signal, the same
Archive — just a different artifact id.

---

## §3.2 GEPA from scratch

GEPA (Agrawal et al., ICLR 2026 Oral) is *population-based reflective
mutation with Pareto selection*. Three distinctive ideas:

1. **A population** of N candidates evolves together. Each generation
   produces offspring; the population is replaced by selection.
2. **Reflective mutation**: instead of "rewrite this prompt to be
   better," the proposer reads a *trajectory* — what the agent did with
   this prompt on a specific question — and proposes an edit targeting
   the failure mode. The LLM critiques what went wrong, then edits.
3. **Pareto selection**: when scoring uses multiple objectives, keep
   the non-dominated front. A candidate that scores higher on quality
   but uses more tokens isn't dominated by a candidate that's better
   on both — both survive.

Run:

```
python chapters/ch03/minimal_gepa.py
```

The script:

- Defines a `Population` as a list of 4 candidate prompts. (~10 lines.)
- Defines `reflective_mutate(parent, trajectory)`: one LLM call where
  the system prompt is "critique what went wrong on this trajectory
  and propose an edit." (~20 lines.)
- Defines a two-objective scoring step: `LiveTrajectoryJudge` for
  quality, `MetricSignal(metric="tokens")` for cost. (~10 lines.)
- Defines `pareto_front(population, objectives)`: keep non-dominated
  candidates. (~15 lines.)
- Runs three generations and prints the Pareto front each time.
  (~25 lines.)

The reader watches the two-objective tradeoff in action: in generation 1
the population has one high-quality / high-cost candidate and one
medium-quality / low-cost candidate. Both survive. Generation 2's
offspring sample from both. By generation 3 the front contains
candidates the reader can pick from depending on their cost tolerance.

The framework version is `helix.search.gepa.GEPA`, which adds caching,
budget enforcement, observability, and a richer crossover operator.

### Pareto, in concrete terms

A candidate is **dominated** if another candidate scores at least as
high on every objective and strictly higher on at least one. The
**Pareto front** is the set of non-dominated candidates: every member
is best at *something*. With objectives `(quality, -tokens)` (negated
because lower tokens is better), the front looks like a staircase:
quality goes up, cost also goes up, and the front is the upper-right
boundary of the population in score space.

This is what GEPA's selection step keeps. Single-objective search keeps
one winner; multi-objective search keeps the staircase.

---

## §3.3 DGM from scratch

DGM (Zhang et al., 2025) is *archive-evolutionary search*. Three
distinctive ideas:

1. **An archive instead of a population.** Every variant ever produced
   stays in the archive forever. There is no generational replacement —
   nothing is discarded.
2. **Quality-diversity sampling.** Selecting the seed for the next
   mutation samples from the *whole archive*, weighting both by score
   (quality) and by some behavioral or content distance from recent
   picks (diversity). A round-20 mutation might fork from a candidate
   produced in round 3 because it's been a long time since anything
   in that branch was tried.
3. **Lineage as a first-class object.** Every mutation carries a parent
   pointer back to its seed. The archive's tree records the full
   evolutionary history; you can replay any branch.

Run:

```
python chapters/ch03/minimal_dgm.py
```

The script:

- Defines an `Archive` as a list of `(artifact, score, parent_id)`
  tuples. (~15 lines.)
- Defines `sample_with_quality_diversity(archive, k)`: weight by score,
  penalize recent picks. (~20 lines.)
- Defines `mutate(parent)`: one LLM call to rewrite, conditioned on
  the parent's score and feedback. (~15 lines.)
- Runs ~10 rounds and prints which round/parent the sampler picked each
  time. (~30 lines.)

The reader sees that round 10 didn't fork from round 9's winner — it
forked from round 3, because round 3 was high-scoring and the sampler
hadn't picked from that branch lately. That's the DGM dynamic: history
is alive, not just the latest generation.

The framework version is `helix.search.dgm.DGMSearch`. It accepts a
**pluggable mutator** so you can use blind LLM mutation, SPO-style
feedback-conditioned mutation, or GEPA's reflective mutation as the
operator inside DGM's archive loop. Same search shape, different
mutation strategies.

### What DGM is, what DGM isn't

DGM's paper applies the pattern to code (the agent edits its own
implementation, runs SWE-Bench, gets better at coding). That result
costs $22K. The chapter applies the same *pattern* to a tool description,
costing ~$1 per chapter run.

The chapter is honest about this: the pattern generalizes, the
benchmark numbers don't. You won't see a 20→50% lift on a coding
benchmark from improving a tool description; you'll see a measurable
quality improvement on the eval set, at a budget that lets you actually
try it.

The code-level DGM, where the artifact under search is the tool's
implementation and not just its description, is Ch 11/12 territory.
The same `DGMSearch` class will be reused there.

---

## §3.4 Three methods on one artifact

SPO mutates one candidate at a time and accepts on win. GEPA evolves a
population with reflection and Pareto selection. DGM searches an archive
with quality-diversity sampling. They are different strategies for the
same problem: find a better artifact than the one you have.

The multi-improver pattern (SPEC §16.1) lets all three run concurrently
on the same artifact. Each writes to the shared archive. `archive.best()`
returns the overall winner regardless of which method produced it.

Two runnable demonstrations:

**Parallel: three improvers, same artifact**

```
python chapters/ch03/dual_improver.py
```

(The script name predates the chapter's expansion to three methods; the
file now runs SPO + GEPA + DGM in parallel rounds on the tool description.
The shape is identical to running two improvers; the framework treats
each `OfflineImprover` independently and the archive arbitrates by score.)

Each method has its own per-round budget. The reader sees three columns
of round-by-round output, the final archive contains candidates from
all three methods, and the best candidate may have come from any of them.

**Sequential: escalating strategy chain**

```
python chapters/ch03/escalating_improver.py
```

A `StrategyChain` runs SPO until it fails N times consecutively, then
rotates to GEPA, then to DGM. The framework retires methods that
plateau and brings in the next strategy. The cost shape is: cheap
(SPO) → moderate (GEPA, more candidates per round) → thorough (DGM,
sampling from full history).

Both scripts use a two-objective signal: `LiveTrajectoryJudge` weighted
0.7, `MetricSignal(metric="tokens")` weighted 0.3. The Pareto-aware
methods (GEPA) keep the front; the others use the weighted average
directly.

After running either script, launch the dashboard:

```
streamlit run helix/dashboard/app.py
```

The artifact lineage view shows the three methods' candidates as three
colored branches. The score timeline shows where each method
contributed. The promote-to-live button works the same as in Ch 2: a
human reviews and promotes the overall winner.

---

## Cost expectations

Practical-budget recipes for one full chapter run:

| Section | Per run | Notes |
|---------|---------|-------|
| §3.1 warmup | ~$0.30 | 3 SPO rounds on tool description, small eval slice |
| §3.2 GEPA from scratch | ~$1.00 | Population of 4, three generations, two-objective scoring |
| §3.3 DGM from scratch | ~$1.00 | 10 rounds, three eval questions per round |
| §3.4 parallel three-method | ~$2.00 | Three improvers, three rounds each |
| §3.4 escalating chain | ~$1.00 | Five rounds total across rotating methods |
| **Full chapter** | **~$5.00** | All scripts run once |

Iterate cheaply by reducing the eval slice (`QUESTIONS_PER_ROUND`) and
the rounds/generations. Most chapter knobs are at the top of each
script for exactly this purpose.

---

## What's deferred

- **Code-level DGM**: aiming `DGMSearch` at a tool's implementation
  rather than its description. Chapter 11/12, where sandboxed code
  execution joins the framework.
- **Memory-grounded search (MemRL)**: evolutionary search where the
  artifact is a memory entry, not a prompt. Chapter 4.
- **Online evolutionary search**: continuous mutation against live
  traffic, no labeled eval set. Chapter 8 (with HITL and feedback).
- **Combining evolutionary search with reflection as a first-class
  Signal**: a Reflector that critiques agent behavior and feeds its
  output to *any* search method. Chapter 5.
