# Persona-rubric optimization (design)

The substrate for the SPO -> GEPA -> DGM comparison that finally puts the methods
in the regime they were built for. The agent plans trips for users drawn from a
set of hidden personas, and a coarse rubric judge rates each trip and returns one
vague sentence of feedback. Because different personas reward different trips,
there is no static answer to copy into the prompt; the optimum is a generalizing
strategy, which is what a search should be made to find.

## Why this is the right regime

Every earlier substrate let a capable proposer win with SPO, for one of two
reasons: the feedback named the fix, or a single hidden answer could be inferred.
Multiple personas remove both. A prompt that hard-codes one taste earns five stars
from one persona and two from another, so the optimum cannot be a preference list;
it has to be a policy for acquiring and applying preferences. That shifts the
target from knowledge to strategy, which is the only thing worth a search.

This is also the regime where GEPA and DGM have real structural advantages over
SPO, statable precisely:

- **No copy shortcut.** The coarse feedback is one user's reaction; baking it in
  hurts the others, so there is nothing to paste.
- **Multi-modal, noisy objective.** The round score averages different personas
  that reward different things, sampled noisily. SPO accepts or rejects on a single
  noisy average and is pulled toward whichever persona dominated the sample; a
  population keeps a Pareto front and an archive never discards on one unlucky read.
- **Crossover has something to combine.** One candidate learns families, another
  learns luxury; merging them is what GEPA's crossover is for and what a single line
  cannot do.

## The persona model (shared by both tiers)

A small set of personas, each a preference profile over the dimensions the trip
exposes. Five is enough to be multi-modal:

- **Family.** Mid budget, comfortable hotel (rating 8+, a pool is a plus),
  activities outdoor / landmark / museum, no nightlife, relaxed pace (about two
  activities), economy nonstop.
- **Luxury.** High budget, top-rated hotel (9+), fine dining / museum / nightlife,
  packed pace, business cabin nonstop.
- **Solo traveler.** Lower-to-mid budget, a modest well-located hotel where rating
  matters little, an exploratory mix of activities (museum / outdoor / nightlife /
  landmark), packed pace, economy and a stop is fine if cheaper.
- **Foodie couple.** Mid-to-high budget, comfortable hotel, food-heavy with some
  culture, moderate pace, economy nonstop.
- **Business.** High budget, convenient comfortable hotel, minimal activities
  (zero or one), efficient, nonstop.

Multi-modality is the point: a luxury hotel plus nightlife delights Luxury and
fails Family on both the nightlife penalty and the budget, while a cheap modest
hotel plus a varied, packed activity slate delights the Solo traveler and fails
Luxury, so no single trip maxes the set. The policy must adapt per persona, which
is exactly what defeats a static answer.

## The signal: a coarse rubric judge

`PersonaRubricJudge` scores a reconstructed trip against the active persona and
returns a 1-to-5 rating (kept as a [0,1] fraction internally) plus one vague
sentence. The score averages per-dimension match (hotel, activities, flight,
budget), each a persona-specific function, so the rating is a smooth scalar with a
gradient. The feedback names only the worst-matched dimension and only as a vibe
("the hotel did not suit me", "too many activities for my taste", "the activities
were not my style"), never the target value, so it cannot be copied into the prompt.

For a book-reproducible experiment the score is a deterministic function of the
trip against the persona's preference vector; in production it would be an LLM
judge. A **noise knob** adds optional jitter to the score (and occasionally a
misleading feedback line) so we can show the robustness axis: at zero noise SPO may
keep up, and as noise rises the population and archive methods should pull ahead.

## Tier 1: cues in the request (build and gate first)

The request signals the persona implicitly ("a 2-day New York trip for my family
with a hotel and a few activities", "a luxe New York getaway for me", "a cheap
backpacking weekend in New York"). The agent must learn to map the cue to a
preference profile and tailor the trip. The genesis policy is a generic planner
that books without tailoring; the oracle policy encodes the cue-to-preference
mapping for all five personas; the search must discover that multi-branch mapping.

This favors GEPA and DGM because the mapping is multi-branch: SPO improving the
family branch need not touch the luxury branch, while a population develops
different branches and crossover combines them, and the coarse per-task feedback
reveals only one persona's dissatisfaction at a time. Tier 1 builds on the agent we
already have, with no user simulator, so it is the cheaper, faster read on whether
the ladder fires.

## Tier 2: elicitation (escalate to if Tier 1 shows the ladder starting)

The request is deliberately vague ("plan me a 2-day New York trip with a hotel and
a few activities") and the persona is hidden, so there is no cue to exploit. An
`ask_user(question)` tool, backed by a persona simulator that answers in character,
lets a clarifying question live inside the existing tool loop. The genesis policy
guesses and books; the optimal policy discovers that it should ask about who is
travelling, budget, and pace before booking, then tailor. Watching the optimizer
invent the questioning behavior, rather than us scripting it, is the demonstration.

The persona simulator answers deterministically by mapping the question's topic to
the persona's profile (vague question, vague answer), so the multi-turn loop stays
reproducible. Tier 2 is the strongest case for GEPA because the optimum is a genuine
interaction strategy that is high-dimensional (which questions, how to use the
answers, how to trade off when answers conflict) and only learnable through the
coarse reward.

## Generalization: train and test personas/requests

Because the objective is an average over personas, a policy could overfit to the
exact personas and requests in the eval set. The experiment uses a train/test
split: optimize on a training set of personas and requests, then report the final
policy's rating on held-out requests (new cities and dates for the same personas)
and, ideally, a held-out persona. A memorized preference list will not generalize;
an adaptive or elicitation strategy will, so the split both keeps the experiment
honest and is where the more powerful methods should show a durable edge.

## What is optimized and how it wires in

- **Artifact:** the system prompt (`prompt.travel.system`, a PROMPT artifact),
  already targetable through `build_travel_agent(system_prompt=...)`.
- **Search:** SPO, GEPA, DGM on the same artifact and task set, exactly as in the
  compliance runs, so the only new pieces are the signal and the tasks (and, for
  Tier 2, the `ask_user` tool and simulator).
- **Eval:** a multi-persona eval set; each round runs the agent across the training
  personas and averages the rating. The judge is deterministic, so only the agent
  passes cost LLM calls (Tier 1), plus the elicitation turns in Tier 2.

## Gating plan (same discipline as before)

1. **Deterministic headroom (no LLM).** For each persona, brute-force the
   best-achievable rating and a generic trip's rating, confirm the spread, and
   confirm no single trip maxes all personas (multi-modality).
2. **Oracle A/B (sonnet).** Genesis policy vs the oracle (Tier 1 mapping, Tier 2
   elicitation) across personas; confirm the oracle clears genesis by a wide
   margin so the headroom is real and reachable.
3. **SPO-plateau gate (the decisive test).** Does SPO climb and stall below the
   oracle on the multi-modal objective? If yes, the ladder is justified.
4. **Escalation.** Run GEPA and DGM, with and without judge noise, and report the
   train and held-out ratings for each method.

If SPO climbs to the ceiling even here, the conclusion is settled: text-prompt
search does not separate these methods, and DGM's showcase moves to the
self-improving-code chapters. If SPO plateaus while GEPA and DGM climb, we have the
honest demonstration, in the regime the methods were designed for.

## Build list

1. `agents/travel_persona.py`: the `PreferenceProfile` and `PERSONAS`,
   `score_persona(trip, persona) -> (rating, coarse_feedback)`, `PersonaRubricJudge`
   with the noise knob, the genesis and oracle policies (Tier 1 mapping; Tier 2
   elicitation), and (Tier 2) the `ask_user` tool, persona simulator, and a
   `build_persona_agent` that adds the tool.
2. `chapters/ch03/travel_persona_tasks.json`: the Tier 1 cue requests and Tier 2
   vague requests, with a train/test split.
3. Gating scripts mirroring the compliance ones (deterministic check, oracle A/B,
   SPO plateau, escalation), reused for both tiers.
4. Tests for the persona scoring and the multi-modality (deterministic, no LLM).

Reused as-is: the simulation tables, the agent loop, `reconstruct_trip`, SPO, GEPA,
DGMSearch, OfflineImprover, and the gating pattern.

## Open questions for the author

- Five personas or more, and whether to hold one out entirely for the generalization
  test.
- The weights across hotel / activities / flight / budget in the rubric, set by the
  oracle A/B so genesis reliably underperforms.
- The default noise level for the headline run, and whether noise is a separate
  reported axis (SPO vs GEPA/DGM as noise rises).
- Whether Tier 2's persona simulator stays deterministic or uses a small LLM for
  more natural answers (cost and reproducibility trade-off).
