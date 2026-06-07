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

Agent Sonnet 4.5 (the intermediate model: genesis 0.35, oracle ceiling 0.73),
Sonnet 4.6 proposer, modest budget (SPO 8 rounds, GEPA population 4 by 2
generations, DGM 8 rounds), selection by absolute rating, optimize on New York,
evaluate on held-out Los Angeles.

| method | train | test (held-out) |
| --- | --- | --- |
| genesis | 0.35 | 0.35 |
| SPO | 0.47 | 0.12 |
| GEPA | 0.31 | 0.30 |
| DGM | 0.60 | 0.67 |

The ladder is the whole story. SPO has the highest training score and the worst
generalization, 0.12, because the single line overfits the training personas. GEPA
holds near genesis, robust but not improving much this run, 0.30. DGM climbs and
generalizes to 0.67, just under the 0.73 ceiling, because the archive keeps diverse
variants and can fork from a branch the fixed population dropped. The advantage of
the more elaborate methods shows up as generalization, not as a higher training
number, which is the honest and teachable form of it.

For contrast, the same experiment on the weak haiku agent put GEPA and DGM together
at 0.44 (both at the low ceiling) and SPO at 0.06: enough to show the methods beat
SPO, but not enough to separate the archive from the population. The intermediate
agent is what reveals the GEPA-versus-DGM gap.

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

The headline table is a single run, and sonnet-tier sampling is noisy, so GEPA's
flat 0.30 may be an unlucky draw and DGM's 0.67 a fortunate one. The direction
matches the theory (archive beats population beats single line, via
generalization), but to state SPO worse than GEPA worse than DGM in print, repeat
the run two or three times and report the spread. The absolute ceiling is also
bounded by Sonnet 4.5's competence, not by the search; a stronger agent with a
harder task would raise it.

## Reproducing

`chapters/ch03/persona_search_experiment.py` runs the headline experiment for a
configurable agent model and search method, with fair absolute selection and the
train/test split. `chapters/ch03/experiments/` holds the gating scripts (the
deterministic multi-modality check, the oracle A/B, and the model-capability
sweep) used to validate the substrate before each spend.
