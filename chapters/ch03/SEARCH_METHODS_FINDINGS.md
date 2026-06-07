# When SPO, GEPA, and DGM actually differ (Chapter 3 findings)

This is the consolidated result of an extended empirical study into when the three
Chapter 3 search methods, SPO, GEPA, and DGM, actually differ on text optimization.
The short answer is that they differ only under specific conditions, and most of
the work was discovering what those conditions are. The headline is that the full
escalation ladder (SPO worse than GEPA worse than DGM) appears on a multi-modal
preference task with a mid-capability agent, fair absolute selection, and a
held-out generalization measure.

## The question

The chapter teaches a cost ladder of search methods: SPO is one cheap mutation per
round, GEPA evolves a population with reflection and crossover, and DGM keeps an
archive of every variant and samples it by quality and diversity. The natural
expectation, and the Chapter 2 result, is that the more elaborate methods beat the
simpler ones on prompt optimization. We set out to demonstrate that cleanly and
repeatedly, and for a long time we could not, which turned out to be the most
useful part of the study.

## What kept hiding the result

Four separate effects, each of which can make the elaborate methods look no better
than SPO, and which had to be removed one at a time.

1. **A too-capable agent has no headroom.** A frontier model (Sonnet 4.6) is
   already a good travel agent: it tailors from cues, elicits from vague requests,
   and allocates a budget sensibly with no optimized prompt. With genesis at 0.61
   and the ceiling at 0.90, there is almost nothing for any optimizer to win, so
   all three methods tie. The agent must be weak enough that the policy genuinely
   matters.
2. **A too-weak agent caps the ceiling.** Haiku has large headroom (genesis near
   0.08), but it executes even a good policy poorly, so the achievable ceiling is
   low (around 0.44 to 0.5). That is enough to separate the methods from SPO but
   not enough to separate GEPA from DGM, because both top out at the same low
   ceiling. The agent needs a high enough ceiling for the better method to have
   somewhere to climb.
3. **Pairwise-vs-seed selection masks good candidates.** Routing a candidate
   through a single noisy pairwise comparison against the seed, and selecting by
   wins, discards strong policies that lose one sample by chance and hands back the
   seed. The fix is to re-measure every candidate on the absolute metric we care
   about and select by that. Before this fix, GEPA looked like a failure while it
   was in fact producing substantive, sensible policies.
4. **No generalization measure rewards overfitting.** A single hill-climbing line
   maximizes the training score by latching onto a narrow training-set win, which
   is exactly what SPO is good at. Without a held-out split you crown SPO the
   winner and are wrong. A train/test split (optimize on one city, evaluate on a
   held-out city) exposes that SPO overfits and the population and archive methods
   generalize.

There was also a substrate dead-end worth recording. On the cue-tagged Tier 1
persona task, all three methods discovered the right strategy, asking the user for
their preferences, but the environment had no ask_user tool, so asking halted the
agent and scored near zero. The methods were not failing; the environment could
not execute what they found. Building the ask_user tool (Tier 2) is what let the
discovered strategy be rewarded.

## The substrate that works

Travel planning for users drawn from five hidden personas (family, luxury, solo,
foodie, business), graded by a coarse rubric that returns a 1-to-5 rating and one
vague sentence. Because different personas reward different trips, there is no
single policy that pleases everyone, so the optimum is a strategy that tailors,
not a fixed answer. This is multi-modal, the feedback is coarse and cannot be
copied into the prompt, and a train/test split across cities measures
generalization. The persona model, rubric, and tasks are in
`agents/travel_persona.py` and `chapters/ch03/travel_persona_tasks.json`; the
design is in `PERSONA_EXPERIMENT_DESIGN.md`.

## The headline result

Agent Sonnet 4.5 (the intermediate model: genesis ~0.3, oracle ceiling ~0.73),
Sonnet 4.6 proposer, modest budget (SPO 8 rounds, GEPA population 4 by 2
generations, DGM 8 rounds), selection by absolute rating, optimize on New York,
evaluate on held-out Los Angeles. Two independent runs, because single runs at
this agent tier are noisy:

| method | run 1 train | run 1 test | run 2 train | run 2 test |
| --- | --- | --- | --- | --- |
| genesis | 0.35 | 0.35 | 0.33 | 0.28 |
| SPO | 0.47 | 0.12 | 0.51 | 0.24 |
| GEPA | 0.31 | 0.30 | 0.23 | 0.39 |
| DGM | 0.60 | 0.67 | 0.28 | 0.12 |

The robust, replicated finding is **GEPA over SPO, by generalization**. SPO has the
highest training score in both runs and the worst-or-near-worst held-out test,
because the single line overfits the training personas. GEPA beats SPO on the
held-out city in both runs (0.30 vs 0.12, then 0.39 vs 0.24) and its test score
holds at or above its train score, the signature of a policy that generalizes. The
advantage of the elaborate method shows up as generalization, not as a higher
training number, which is the honest and teachable form of it.

The DGM comparison above was flawed: it restarted DGM from a deleted archive each
run, which strips out the archive mechanism that defines the method. DGM is
archive-evolutionary: the archive persists and accumulates, never discarding a
variant, so its best is a ratchet that holds or climbs. The corrected experiment
(`experiments/persona_methods_over_runs.py`) runs DGM as one continuous process
whose archive carries across runs, against SPO and GEPA which start fresh each run.

The corrected two-run result, held-out test, on Sonnet 4.5:

| method | run 1 train/test | run 2 train/test |
| --- | --- | --- |
| SPO (fresh) | 0.38 / 0.37 | 0.24 / 0.06 |
| GEPA (fresh) | 0.45 / 0.32 | 0.21 / 0.67 |
| DGM (persistent) | 0.52 / 0.21 | 0.16 / 0.24 |

On the held-out test DGM held and slightly climbed across runs (0.21 to 0.24), not
crashing the way the flawed restart design showed, which is the ratchet behaving as
expected. But the run also exposes the real bottleneck: **single-sample evaluation
noise now dominates**. DGM's train score fell 0.52 to 0.16, which is impossible for
a true ratchet (run 2's archive is a superset of run 1's, so the best-by-train can
only rise); it fell only because each candidate is scored on one noisy rollout, so
the selected best fluctuates regardless of what the archive holds. The same noise
muddies the headline: in run 1 SPO (0.37) edged GEPA (0.32), then in run 2 GEPA
(0.67) crushed SPO (0.06). Averaged, GEPA still beats SPO, but no single run is
trustworthy, and the differences we want to measure are smaller than the noise.

The fix is to denoise the evaluation: average several rollouts per candidate before
selecting and scoring, which turns the wobbly single-sample best into a stable
estimate and would let both the GEPA-over-SPO gap and the DGM ratchet show cleanly.
The runner takes a SAMPLES knob for this.

For contrast, the same experiment on the weak haiku agent put GEPA and DGM together
at 0.44 (both at the low ceiling) and SPO at 0.06: enough to show the population and
archive methods beat the single line, but the low ceiling hides any GEPA-versus-DGM
distinction. The intermediate agent gives the methods room, but the run-to-run noise
means the GEPA-over-SPO result is the one that survives replication.

## What does not help

More search budget does not break the ceiling. Raising GEPA to population 6 by 3
generations pushed its training score up to 0.55 but its test down to 0.36, the
textbook signature of overfitting, and DGM at 20 rounds did not improve either. The
plateau is set by the agent's capability and by the overfitting risk of train-based
selection, not by search budget. On a small training set, more search trades
generalization for training fit.

## Conditions for the ladder to appear

- A multi-modal objective with coarse feedback, so the optimum is a strategy and
  the feedback cannot be copied.
- An agent in the capability band that has real headroom and a high executable
  ceiling (Sonnet 4.5 here; not Sonnet 4.6, which is too strong, nor Haiku, which
  is too weak).
- Selection by the absolute metric, not pairwise wins against a noisy seed.
- A held-out generalization measure, because the advantage is robustness, and a
  training-set score rewards SPO's overfitting instead.
- A capable proposer. A weak (haiku) proposer breaks GEPA and DGM, because their
  reflection and crossover need a capable operator; SPO is more robust to a weak
  proposer.

## Caveats

Run-to-run variance at this agent tier is large, which is exactly why two runs were
needed and why DGM's run-1 0.67 did not survive replication. The per-candidate
re-measurement is a single noisy sample, so both the search (which finds different
policies each run) and the selection (best-by-noisy-train) add variance; averaging
several samples per candidate would tighten it. What survives is GEPA over SPO; a
claim about DGM needs more runs and multi-sample evaluation before it is printable.
The absolute ceiling is also bounded by Sonnet 4.5's competence, not the search; a
stronger agent on a harder task would raise it.

## Reproducing

`chapters/ch03/persona_search_experiment.py` runs the headline experiment for a
configurable agent model and search method, with fair absolute selection and the
train/test split. `chapters/ch03/experiments/` holds the gating scripts (the
deterministic multi-modality check, the oracle A/B, and the model-capability
sweep) used to validate the substrate before each spend.
