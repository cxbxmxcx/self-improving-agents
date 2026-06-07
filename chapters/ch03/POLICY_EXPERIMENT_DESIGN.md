# Chapter 3 policy-optimization experiment (design)

This is the design for the lead Chapter 3 example: GEPA and DGM optimizing the
travel agent's *policy* (its system prompt) rather than a tool description. The
tool-description experiment becomes a shorter sidebar on why optimizing thin text
is subtle; this is the example that motivates reaching for a search method at all.

## Why the artifact changes from description to policy

A tool description is the thinnest optimizable artifact in an agent, because the
parameters already live in the function schema the model sees. We had to
manufacture three hidden, prior-neutral gotchas to give description search any
leverage, which is a sign the example teaches the method instead of motivating it.

A policy has genuine strategic depth that a capable model does not get for free.
GEPA in the literature optimizes instructions and policy, and DGM optimizes the
agent's own tools and code; neither is at its best rewriting help text. Pointing
both at the policy aligns the example with what the methods are actually for.

## Where the leverage comes from (three structural gaps, no hidden facts)

The headroom must be strategic, not a buried value, or we repeat the description
problem. Three gaps create durable headroom even for a strong model.

1. **Plan-then-act versus greedy-act.** The booking flow is sequential, which
   biases the agent to commit to each component before seeing the whole picture.
   The global optimum requires searching every component first and allocating
   before booking, which a naive agent does not do on its own.
2. **Budget allocation (a small knapsack).** Maximizing trip quality subject to a
   single total budget across flight, hotel, and activities is a global tradeoff.
   Greedy-cheapest leaves quality on the table and greedy-best busts the budget,
   so neither default behavior is optimal.
3. **Recovery and relaxation.** Some tasks are over-constrained on a soft
   preference, so a naive policy hits an empty search and gives up, scoring zero
   on a required component. A policy that relaxes the lowest-value preference and
   retries completes the trip.

None of these is a hidden number the description had to reveal. They are strategy,
which is exactly what justifies a search over policies.

## The objective and signal (graded, so DGM has a gradient)

A new deterministic signal, TripPlanJudge, scores the reconstructed trip against
each task's objective and returns both an absolute score and a pairwise preference
(so GEPA's pairwise machinery works, exactly as TravelTaskJudge does). The score
is graded in [0, 1] so the reward surface has a gradient, which our earlier
GEPA-versus-DGM finding showed DGM needs.

The score combines three terms, with required-component hard gates:

- **Completeness C.** Each required component (flight on the right route and
  dates, hotel in the right city for the right nights, at least the minimum
  activities) contributes its share; a missing or wrong-field required component
  scores zero for that share.
- **Budget factor B.** `total_cost <= budget` gives `B = 1`; over budget falls off
  linearly to `B = 0` at `1.5 x budget`. It multiplies the rest, so busting the
  budget is heavily but not infinitely penalized, preserving a gradient.
- **Quality Q.** Within budget, a normalized sum of nonstop bonus, hotel rating,
  activities up to a cap, and category matches, divided by the task's achievable
  maximum so Q is in [0, 1].

`score = B * (w_c * C + w_q * Q)`. A greedy-cheapest policy completes the trip but
scores low Q; a greedy-best policy busts B; the optimum spends the budget to
maximize Q, which is the strategy the search must find.

## Honest pricing (isolate strategic leverage from the gotchas)

To keep the policy leverage purely strategic, the policy example uses the good
(oracle) tool descriptions produced by the sidebar experiment, so the agent
already passes `cabin='economy'` and `rate_code='Q'`. Pricing is then honest, with
no hidden multiplier, and the budget pressure comes entirely from real allocation
across components. This also composes the two sections: the sidebar produces good
descriptions, and the lead example fixes them and optimizes the policy on top.

## Task suite (whole-trip, deterministic, ~6-8 tasks)

Each task carries a destination and dates, a total budget, required components,
and weighted soft preferences. The suite mixes budget-tight tasks (force
allocation) with over-constrained tasks (force relaxation).

Worked example. "Plan a 3-night New York trip from SFO, total budget $1200; I want
a nonstop flight, a four-star-or-better hotel, and two food activities."

- Required: flight SFO to JFK, hotel JFK for 3 nights, at least 2 activities.
- Greedy-best books FL101 ($412) plus HT201 (9.2) at $410 x 3 = $1230, already
  $1642 and over budget, so B collapses.
- Greedy-cheapest books a one-stop flight and HT205 (6.4), cheap but low Q.
- The optimum books FL101 nonstop ($412), HT204 (8.2) at $246 x 3 = $738, and two
  food activities (about $45), totalling roughly $1195 under $1200, with a nonstop
  flight, an 8.2 hotel, and two food activities, so Q is high. Reaching it requires
  reserving the hotel cost and fitting the rest, which a greedy agent misses.

## Genesis policy and an example discovered policy

Genesis (deliberately strategy-free): "You are a travel assistant. Use the tools
to book the trip the user asks for." It gives no guidance on planning, allocation,
ordering, or recovery, leaving the headroom open.

A strong policy the search might discover: "Before booking anything, search every
component and note prices and quality. Compute your total budget envelope and
reserve the hotel cost (nightly rate times nights) first. Spend the remainder to
maximize quality: a nonstop flight, the highest-rated hotel you can afford, and as
many preferred activities as fit. If you cannot satisfy every preference within
budget, drop the lowest-value one rather than exceed the budget or skip a required
component. Book only once the full plan fits, then confirm."

## Wiring (reuses the existing harness)

- **Target artifact:** `SYSTEM_PROMPT_ID = "prompt.travel.system"` (already a
  `Subtype.PROMPT` genesis artifact). GEPA and DGM target it directly; the agent
  clones via `build_travel_agent(system_prompt=candidate)` through `with_artifacts`.
- **Single improver.** One policy artifact means one improver, so no multi-improver
  pattern, no per-tool archives, and no cross-seed concern.
- **Whole-trip eval.** The eval set is the task suite; the policy is global, so
  whole-trip scoring is now the correct unit, not a confound.
- **Signal:** TripPlanJudge over `reconstruct_trip`, returning absolute score and
  preference.
- **Search:** GEPA (population) and DGM (archive) on the same artifact and suite,
  re-running the head-to-head on a graded surface. DGM should fare better here than
  on the flat activity gotcha, a useful callback to the earlier finding.

## De-risk before any spend (the lesson from the descriptions)

Validate headroom with a cheap oracle A/B before running any search. Hand-write
the strong policy above, then score genesis versus oracle on the task suite with
the deterministic signal and a sonnet agent, about eight runs each. If oracle
clearly beats genesis, the headroom is real and GEPA and DGM are worth running; if
not, tune task difficulty (tighten budgets, add over-constrained tasks) until
genesis reliably underperforms. The headroom comes from task hardness, not a hidden
fact, so tuning difficulty is legitimate rather than contrived.

## Build list (most pieces already exist)

1. Keep a deliberately strategy-free genesis policy (trim `DEFAULT_SYSTEM_PROMPT`
   for this example, or add a `GENESIS_POLICY` constant).
2. Add `TripPlanJudge` plus the graded scoring helpers (extends the existing
   scoring in `agents/travel.py`).
3. Author `travel_policy_tasks.json` (the whole-trip task suite with budgets,
   required components, and preferences).
4. Add `policy_optimization.py` (GEPA) and reuse it for DGM via a flag or a
   parallel script, mirroring the tool-description runners.
5. Tests for the objective and the headroom (deterministic, no LLM).

Reused as-is: the simulation tables, the agent loop, `reconstruct_trip`, GEPA,
DGMSearch, OfflineImprover, the console renderer, and the oracle-A/B pattern.

## Spec alignment

Optimizing the system prompt is the canonical GEPA use, PROMPT is the L1 artifact,
and the agent loop does not change; improvement is still a search over an artifact
recorded in the archive with lineage. The policy artifact also recurs across later
chapters, so gains here compound rather than staying local to one tool.

## Open questions for the author

- Budget and quality weights (`w_c`, `w_q`) and the over-budget falloff shape; set
  by the oracle A/B so genesis reliably underperforms.
- Whether to keep both GEPA and DGM in the lead example or lead with one and revisit
  the other in the head-to-head sidebar.
- Whether the policy example fixes the oracle descriptions (cleanest) or keeps the
  genesis descriptions and lets the policy also learn to use economy and Q.
