# Chapter 3: Searching for better text with evolutionary methods

Chapter 2 introduced one search method (SPO), one artifact (the system
prompt), and one signal family (the LLM-as-judge) on a RAG agent. Chapter
3 widens all three. The running example becomes a **travel task agent**: a
multi-tool agent that books flights, hotels, and activities against a
deterministic simulation. With several tools, the natural-language *tool
description* is what tells the model which tool to call and which
constraints to pass, so optimizing it produces a visible jump in task
success, which a single-tool RAG agent cannot show. And because the trip is
checkable, the signal becomes **ground truth**, not a judge: a
deterministic task-success score that evolutionary search (GEPA) is best
driven by. The reader learns GEPA and DGM from scratch, applies them to a
tool description, and runs the framework's multi-improver pattern on the
same artifact.

Four sections, runnable scripts:

| § | Concept | Script |
|---|---------|--------|
| 3.1 | Evolutionary economics, tool descriptions as artifacts, the task agent + ground-truth signal | `travel_tool_optimization.py` |
| 3.2 | GEPA from scratch: population, reflective mutation, Pareto over two objectives | `gepa_travel_from_scratch.py` |
| 3.3 | DGM from scratch: archive of variants, quality-diversity sampling | `dgm_travel_from_scratch.py` |
| 3.4 | Several methods on one artifact: SPO + GEPA + DGM via the multi-improver pattern | `travel_tool_optimization.py`, `escalating_travel.py` |

The simulation, agent, and signal live in `agents/travel_sim.py`,
`agents/travel.py`, and `helix/signals/task_success.py`; the scenarios are
`chapters/ch03/travel_scenarios.json`. The earlier RAG-based scripts
(`tool_description_loop.py`, `minimal_gepa.py`, `minimal_dgm.py`,
`dual_improver.py`, `escalating_improver.py`) remain in the directory as the
prior, judge-based variants for comparison.

## The search-method cost ladder (revenue task)

The chapter's headline demonstration is the cost ladder: SPO, GEPA, and
DGM run as the identical improvement loop on one deterministic task,
swapping only the Search object, and score 0.63 < 0.86 < 1.00. The task is
a revenue data-investigation agent (`agents/revenue.py`) with hidden
business rules and plain-Python scoring, so the gap between methods is
mechanical and inspectable, with no LLM judge in the loop.

| Script | Role |
|--------|------|
| `revenue_check.py` | Validates the substrate: genesis misses the gotchas, the oracle scores 1.0, both deterministic |
| `revenue_spo_loop.py` | SPO plateaus at 0.63 on outcome-only feedback |
| `revenue_gepa_loop.py` | GEPA's trace reflection reaches 0.86, then the population converges |
| `revenue_dgm_loop.py` | DGM's quality-diversity archive recovers the last rule and hits 1.00 |

The full findings, including why the configuration matters (mid-tier
proposer, interfering rules, smooth scoring), are written up in
[`REVENUE_SEARCH_LADDER.md`](REVENUE_SEARCH_LADDER.md). Read that before
re-running; the result is conditional on the recipe, not a universal
ranking of the methods.

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
from running `05_spo_offline_loop.py` at least once — Ch 3 inherits Ch 2's
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

### The task agent and the ground-truth signal

The travel agent's `search_flights` tool accepts `nonstop` and `max_price`,
but its genesis description ("Search for flights.") never mentions them, so
the agent ignores those constraints and books the wrong flight. That is the
failure mode the chapter fixes by searching over the description.

Because a trip is checkable, the signal is ground truth, not a judge.
`helix.signals.task_success.TaskSuccessSignal` runs the agent against each
scenario and scores the booked trip deterministically (right route, date,
nonstop, under budget, the activities asked for), and `TravelTaskJudge`
(`agents/travel.py`) is the pairwise form the framework round consumes. The
itinerary is read back from the trajectory by `reconstruct_trip`, so the
agent stays stateless and clones cleanly under `with_artifacts`.

Run the warmup:

```
python chapters/ch03/travel_tool_optimization.py
```

The script targets `prompt.tool.search_flights.description` with framework
SPO and GEPA, judged by `TravelTaskJudge`. Everything from Ch 2 §2.4 carries
over: the same Improver, Archive, and promotion gate. What changed is the
agent (multi-tool, so the description matters) and the signal (ground truth,
so GEPA has something deterministic to climb).

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
python chapters/ch03/gepa_travel_from_scratch.py
```

The script (nothing imported from `helix.search.gepa`):

- Starts a population from the genesis `search_flights` description plus
  reflective variants of it. (~10 lines.)
- Defines `_reflective_mutate(parent, feedback)`: one LLM call that reads
  which scenarios failed and what the agent booked, then rewrites the
  description to pass the missed constraints. (~20 lines.)
- Scores each candidate with `TaskSuccessSignal` (ground-truth task
  success) on one objective and description length, a stand-in for prompt
  cost, on the other. (~10 lines.)
- Defines `_pareto_front(scored)`: keep candidates not dominated on
  (task-success up, cost down). (~15 lines.)
- Runs three generations and prints the Pareto front each time. (~25
  lines.)

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
python chapters/ch03/dgm_travel_from_scratch.py
```

The script (nothing imported from `helix.search.dgm`):

- Defines an `Archive` of `Entry(artifact, score, parent_version,
  times_sampled)` over candidate `search_flights` descriptions. (~15 lines.)
- Defines `sample_quality_diversity()`: weight by score, halved each time a
  branch is re-sampled. (~10 lines.)
- Defines `mutate(seed)`: one LLM call to rewrite the sampled description,
  conditioned on its task-success score. (~15 lines.)
- Scores each candidate with `TaskSuccessSignal` (ground truth) and runs
  ~8 rounds, printing which version the sampler forked from each time.

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
python chapters/ch03/travel_tool_optimization.py
```

The script attaches three `OfflineImprover`s (SPO, GEPA, DGM) to the travel
agent, all targeting `prompt.tool.search_flights.description` and judged by
`TravelTaskJudge`. The framework treats each independently and the shared
archive arbitrates by score, so the best candidate may have come from any
method.

**Sequential: escalating strategy chain**

```
python chapters/ch03/escalating_travel.py
```

A `StrategyChain` runs SPO until it fails N times consecutively, then
rotates to GEPA, then to DGM. The framework retires methods that
plateau and brings in the next strategy. The cost shape is: cheap
(SPO) → moderate (GEPA, more candidates per round) → thorough (DGM,
sampling from full history).

Both scripts grade with `TravelTaskJudge`, the deterministic ground-truth
signal: it reconstructs the booked trip from each trajectory and prefers the
description that satisfied more constraints. To make it multi-objective
(quality versus token cost), wrap it with a `MetricSignal(metric="tokens")`
in a `CompositeSignal`; the from-scratch scripts show the Pareto idea
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
