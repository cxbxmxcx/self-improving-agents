# Chapter 4: The memory artifact and online improvement

Chapters 2 and 3 improved L1 text artifacts (system prompts, tool
descriptions) offline, with a human gate on promotion. Chapter 4 moves
down a layer to L2 memory and turns the cadence online. Memory entries
become searchable artifacts; the agent learns from its own traffic as it
runs; and two RL-style search methods — MemRL for memory and GRPO-style
group-relative selection for tool calling — join the evolutionary methods
from Ch 3 under one Search abstraction.

The chapter's intellectual claim, locked into the spec at §4.2.1:
**evolutionary search and RL-style search are the same pattern.** Both
propose variants, measure them with a Signal, and select winners. They
differ only in how many candidates per round and in the selection rule.
GRPO is not a fine-tuning method here; it is a selection mechanic applied
external to the model. The "policy update" is artifact promotion, not a
weight update. That keeps Chapter 4 inside the book's strict-external
thesis while still teaching the RL pattern honestly.

Four sections, five runnable scripts:

| § | Concept | Script |
|---|---------|--------|
| 4.1 | Memory entries as searchable artifacts | `memory_as_artifacts.py` |
| 4.2 | MemRL: searching memory for utility | `minimal_memrl.py` |
| 4.3 | GRPO-style group-relative selection for tool calling | `minimal_grpo.py` |
| 4.4 | Online memory learning from live traffic | `online_memory_loop.py` |

This is the heaviest framework chapter so far (~30-40 pages). It adds six
or seven framework modules. Every one is marked below with a FUTURE gap
so the build can be reviewed before it starts.

## The online distinction: Ch 4 vs Ch 8

Both chapters are "online," but they mean different things, and the
difference is the layer rule.

- **Chapter 4 online** is auto memory learning. Memory entries (L2) and
  tool descriptions (L1) are online-safe: the layer rule permits
  auto-promotion because a bad memory entry or a slightly-worse tool
  description does not need a deploy gate. The agent learns from its own
  traffic with no human in the loop.
- **Chapter 8 online** is HITL-gated prompt improvement. System-prompt
  changes (L1, but high-blast-radius) and anything riskier get a human
  review gate, live feedback signals, and the full deploy-review flow.

Chapter 4 is where `OnlineImprover` (the event-driven improver) gets its
foundational demonstration. Chapter 8 adds the human gate on top.

## Shared archive

Memory entries and prompt/tool candidates live in the **same** SQLite
archive, distinguished by `Artifact.kind`. This matches SPEC §4.2's claim
that "the archive is the agent's episodic memory itself." The dashboard
shows memory entries alongside prompt candidates, filtered by kind. There
is one archive per agent, not one per artifact type.

---

## §4.1 Memory entries as searchable artifacts

A memory entry — an episode the agent recalls, a fact it learned, a
tool-selection preference — is an `Artifact(kind=MEMORY_ENTRY)`. It has a
version, a parent pointer, and content. ExpeL's four operators
(ADD / UPVOTE / DOWNVOTE / EDIT) are mutations that produce new versions.

The agent reads relevant entries at request time via a PRE_MODEL hook and
writes new ones at SESSION_END. Both the read and the write go through the
archive, so every memory entry is a searchable artifact that MemRL (§4.2)
can later target.

Run:
```
python chapters/ch04/memory_as_artifacts.py
```

The script runs three requests, writes a memory entry after each, reads
them back on subsequent requests, and prints the archive's `MEMORY_ENTRY`
artifacts with their lineage.

### Framework gaps surfaced by §4.1

```
# FUTURE [gap 1]: helix/memory/episodic.py currently writes entries to its
#   own SQLite store, not to the shared archive as MEMORY_ENTRY artifacts.
#   Need: episodic memory that records entries as searchable archive
#   artifacts (kind=MEMORY_ENTRY) so MemRL can target them. The read path
#   (PRE_MODEL injection) and write path (SESSION_END) both route through
#   the archive.

# FUTURE [gap 2]: helix/memory/operators.py — ExpeL's ADD/UPVOTE/DOWNVOTE/
#   EDIT as named mutations producing new artifact versions with parent
#   pointers. UPVOTE/DOWNVOTE adjust a utility score in metadata; EDIT
#   rewrites content; ADD is genesis; all carry lineage.
```

---

## §4.2 MemRL: searching memory for utility

The artifact under search is the episodic memory entry. The signal is
*utility*: did recalling this entry contribute to a better outcome? The
search reinforces high-utility entries (UPVOTE, keep) and prunes
low-utility ones (DOWNVOTE, evict). This is reinforcement learning
expressed as search — the archive is the memory, the operators are
ExpeL's four, and the selection rule is utility-weighted.

Built from scratch, then bridged to the framework's MemRL search.

Run:
```
python chapters/ch04/minimal_memrl.py
```

The hand-written version (~80 lines): run the agent, attribute the
outcome reward back to the entries it recalled this turn, upvote the ones
that helped and downvote the ones that didn't, repeat. Reader watches a
memory store self-curate over several turns — low-utility entries sink and
get pruned; high-utility entries rise and get recalled more often.

### Framework gaps surfaced by §4.2

```
# FUTURE [gap 3]: helix/signals/utility.py — a UtilitySignal that measures
#   whether a recalled memory entry contributed to a good outcome. Requires
#   reward attribution across the trajectory: the entries recalled at
#   PRE_MODEL share credit for the final outcome. Returns a per-entry score.

# FUTURE [gap 4]: helix/search/memrl.py — MemRL search. The archive IS the
#   memory. propose() yields candidate mutations of memory entries using
#   ExpeL operators; select() keeps high-utility entries. Reuses the
#   archive primitive; the distinctive bit is that the "candidates" are
#   memory entries, not prompts.

# SPEC decision: SignalKind.UTILITY (new) or reuse GROUND_TRUTH? Utility is
#   an environment-derived reward, close to GROUND_TRUTH in shape but
#   semantically distinct (it scores a memory entry's contribution, not an
#   answer's correctness). Leaning new UTILITY member; confirm during build.
```

---

## §4.3 GRPO-style group-relative selection for tool calling

GRPO (Group Relative Policy Optimization, the DeepSeek-R1 method) samples
a group of N responses, scores each, computes each response's advantage
relative to the group mean, and reinforces the above-average ones. In the
paper, "reinforce" means a gradient step on model weights.

This chapter uses GRPO's *selection mechanic* with no weight update. We
sample a group of N tool-calling variants — each from a different prompt
or tool-description artifact — score each on a tool-call-quality signal,
compute each variant's advantage relative to the group mean, and promote
the above-average variants' artifacts. The reinforcement is artifact
promotion. The model is never touched.

This is the chapter's clearest demonstration of the §4.2.1 claim: GRPO,
stripped of its weight-update step, is a Search with a group round shape
and a group-relative selection rule. It sits alongside DGM (archive round,
quality-diversity selection) and GEPA (population round, Pareto selection)
as a sibling, not a different paradigm.

Run:
```
python chapters/ch04/minimal_grpo.py
```

The hand-written version (~90 lines): generate a group of N tool-calling
variants, run each on the eval slice, score tool-call quality, compute
group-relative advantage (`score - group_mean`), select the positive-
advantage variants. Reader sees the group's mean rise round over round as
below-average variants are culled and above-average ones spawn the next
group.

### Framework gaps surfaced by §4.3

```
# FUTURE [gap 5]: helix/improvement/group_round.py — run_group_round(). The
#   N-wise analogue of run_improvement_round (which is pairwise). Sample N
#   variants, run all N against the eval slice, score all N with an absolute
#   signal, compute each variant's advantage relative to the group mean,
#   select the positive-advantage variants. This is a genuinely new round
#   shape — the framework's third, after pairwise and archive-evolutionary.

# FUTURE [gap 6]: helix/search/group_relative.py — GroupRelativeSearch.
#   propose() yields a group of N candidates; select() returns the highest
#   group-relative advantage. Composes with run_group_round.

# FUTURE [gap 7]: a tool-call-quality signal. Did the agent call the right
#   tool with the right arguments? Could reuse LiveTrajectoryJudge with a
#   tool-focused rubric, or add helix/signals/tool_call.py with a dedicated
#   ToolCallSignal that inspects the trajectory's TOOL_CALL steps. Leaning
#   reuse-with-rubric to start; promote to a dedicated signal if it earns it.
```

---

## §4.4 Online memory learning from live traffic

The synthesis. The agent serves simulated live traffic. An `OnlineImprover`
(built in PR 4, SPEC §17) subscribes to SESSION_END. Each completed
trajectory drives two things with no human in the loop:

1. **MemRL updates the memory entries** (§4.2). The entries recalled this
   turn get utility credit from the outcome; low-utility entries are
   pruned, high-utility entries reinforced.
2. **Group-relative selection refines the tool-calling prompt** (§4.3) on
   a rolling window of recent trajectories.

No human gate, because everything here is L2 (memory) or L1
(tool-description) — online-safe by the layer rule. This is the
foundational `OnlineImprover` demonstration. Chapter 8 builds the
human-gated variant on top, for changes that need review.

Run:
```
python chapters/ch04/online_memory_loop.py
```

Replays traffic; shows memory growing and pruning live, and tool-calling
improving via group-relative selection, all without a human promoting
anything. The dashboard's lineage view shows memory entries and tool-
description candidates as separate branches in the same archive.

### Framework gaps surfaced by §4.4

```
# FUTURE [gap 8]: OnlineImprover (exists, PR 4) has only been exercised
#   with a rubric judge on a system prompt. Driving it with MemRL search +
#   UtilitySignal on memory entries, and with GroupRelativeSearch +
#   run_group_round on a tool description, will likely surface integration
#   gaps in the SESSION_END handler and the propose path. Verify both the
#   memory improver and the tool improver run in the event-driven (no
#   driver loop) path.
```

---

## Consolidated framework gap inventory

Chapter 4 forces the framework to grow. This is the full list, the review
artifact before any framework code is written:

| # | Piece | New / modified file | § | Rough LOC | Spec implication |
|---|-------|---------------------|---|-----------|------------------|
| 1 | Memory entries as archive artifacts | `helix/memory/episodic.py` (modify) | 4.1 | ~120 | Confirm §7 memory-is-archive-backed |
| 2 | ExpeL operators as mutations | `helix/memory/operators.py` (new) | 4.1 | ~80 | §1 mutation rule applies to memory |
| 3 | `UtilitySignal` | `helix/signals/utility.py` (new) | 4.2 | ~100 | §3 add SignalKind.UTILITY |
| 4 | `MemRL` search | `helix/search/memrl.py` (new) | 4.2 | ~150 | §4.2 MemRL family |
| 5 | `run_group_round` | `helix/improvement/group_round.py` (new) | 4.3 | ~180 | §4.2.1 group round shape |
| 6 | `GroupRelativeSearch` | `helix/search/group_relative.py` (new) | 4.3 | ~130 | §4.2 GRPO-style family |
| 7 | Tool-call-quality signal | reuse or `helix/signals/tool_call.py` | 4.3 | ~80 | §3 (possibly reuse) |
| 8 | OnlineImprover integration | verify / patch | 4.4 | ~50 | §17 round-shape dispatch |

Total new framework surface: roughly 900 LOC across six or seven modules,
plus tests. This is appropriately the heaviest framework chapter; the
foundation it lays (memory-as-artifact, the group round shape, the RL-as-
search unification) is reused throughout the production half of the book.

### Build order (proposed, after review)

1. Gap 1 + 2 (memory-as-artifact + operators) — foundational; §4.1 needs it.
2. Gap 3 + 4 (UtilitySignal + MemRL) — §4.2.
3. Gap 5 + 6 + 7 (group round + GroupRelativeSearch + tool signal) — §4.3.
4. Gap 8 (OnlineImprover integration) — §4.4 ties it together.

Each step gets tests and a passing full-suite run before the next, same
cadence as Chapters 2 and 3.

---

## Cost expectations

| Section | Per run | Notes |
|---------|---------|-------|
| §4.1 memory as artifacts | ~$0.20 | 3 requests, no improvement loop |
| §4.2 MemRL from scratch | ~$1.00 | utility attribution over several turns |
| §4.3 GRPO from scratch | ~$1.50 | group of N=4 per round, three rounds |
| §4.4 online memory loop | ~$2.00 | replayed traffic with two online improvers |
| **Full chapter** | **~$5.00** | all scripts once |

---

## What's deferred

- **HITL-gated online improvement**: human review, live feedback signals,
  deploy gates for risky L1 changes. Chapter 8.
- **Semantic and procedural memory under search**: this chapter focuses on
  episodic memory. Semantic-fact and skill-library search are later.
- **Metacognitive memory (state that survives the run)**: the planner /
  monitor / reflector scaffold. Chapter 5.
- **Code-level RL-as-search**: GRPO-style selection over tool *implementations*,
  not just descriptions. Chapter 11/12.
