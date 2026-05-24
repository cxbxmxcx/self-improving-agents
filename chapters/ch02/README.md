# Chapter 2: Getting started improving agents

This chapter teaches the smallest end-to-end self-improvement loop: one
prompt artifact, one LLM-as-judge, one search method, one improver. By
the end, the reader can mutate a system prompt offline, score the
mutations against an eval set, and promote a winning version into
production.

Four sections, four runnable scripts:

| § | Concept | Script |
|---|---------|--------|
| 2.1 | The artifact under improvement | `helixagent_v1.py` |
| 2.2 | Measuring with an LLM-as-judge | `minimal_judge.py` |
| 2.3 | Searching with SPO | `minimal_spo.py` |
| 2.4 | The offline improvement loop | `spo_offline_loop.py` |

The chapter assumes the reader has built an agent before. The pre-MEAP
baseline agent (one-shot RAG agent with a fixed prompt) lives in
`chapter_appendices/getting_started/` for readers who want a refresher;
this chapter starts from "you have an agent and you want it to get
better."

Online improvement (auto-promotion, no labeled eval set, rolling
spot-checks) is a Ch 8 topic — it composes with HITL and feedback in a
way that needs both foundations. Evolutionary search (GEPA, multi-search
strategy chains, tool-description improvement) is Ch 3.

## Prerequisites

1. Build the corpus once:
   ```
   python ingestion/build_index.py
   ```
2. Set an LLM provider key. LiteLLM picks up the env var for whichever
   model you choose (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

---

## §2.1 The artifact under improvement

The thing being improved is a single object: the agent's system prompt,
treated as an `Artifact`. Three properties matter:

- **Identity.** The prompt has a stable id (`prompt.helixagent.system`)
  and a version (`1`, `2`, ...). Mutations produce new versions; old
  versions stay readable.
- **Lineage.** Every version after the first carries a parent pointer.
  The archive can reconstruct the full mutation history.
- **The agent reads at request time.** The prompt is not hardcoded in
  the agent's source. The agent reads the live champion from the archive
  on every request.

Run:
```
python chapters/ch02/helixagent_v1.py
```

On first run the archive is seeded with the genesis prompt. The agent
serves whatever the archive currently identifies as the live champion.
Until §2.4 produces a winning candidate and you promote it, that's still
the genesis prompt.

The archive is a SQLite file at `chapters/ch02/runs/helix_archive.sqlite`.
Open it in any sqlite browser to see the `artifacts`, `measurements`, and
`promotions` tables. There's no magic.

---

## §2.2 Measuring with an LLM-as-judge

Before improving, measure. An LLM-as-judge takes (question, candidate
answer A, candidate answer B, reference) and returns LEFT/RIGHT/TIE plus
a feedback string. This section builds one from scratch.

Run:
```
python chapters/ch02/minimal_judge.py
```

The script:

1. Defines a 30-line pairwise judge with one LiteLLM call. The judge's
   system prompt holds the rubric; the user prompt holds the question
   and the two answers.
2. Demonstrates **position bias**: scores the same answer pair twice
   with positions swapped, often getting two different verdicts.
3. Implements **swap-and-agree**: run the judge twice with positions
   swapped, require both runs to agree, default to TIE on disagreement.
   ~15 more lines.

By the end of this section the reader has hand-written exactly what the
framework's `SwapAndAgree(PairwiseJudge(...))` is. The framework version
adds prompt caching, structured Pydantic output, and observability
events; the *algorithm* is what the reader just wrote.

---

## §2.3 Searching with SPO

Now the reader has an artifact and a way to compare two versions of it.
SPO produces the second version.

Run:
```
python chapters/ch02/minimal_spo.py
```

The script:

1. Defines a **mutation** as one LLM call: "rewrite this prompt to do X
   better, here's what the judge said about the last version." ~15 lines.
2. Defines the **SPO loop**: mutate → run candidate on a few eval
   questions → judge candidate vs reference → accept if win, reject if
   loss. Three rounds, accept-on-win hill climbing. ~50 more lines.
3. Prints the per-round verdict so the reader sees three rounds of
   "candidate won" or "candidate lost" with the judge's feedback driving
   the next mutation.

That's SPO. The framework's `helix.search.spo.SPO` is the same algorithm
with caching, budget enforcement, parent pointers, and observability
spans. The reader has internalized the algorithm before seeing the
abstraction.

---

## §2.4 The offline improvement loop

Now the framework version, end-to-end. `OfflineImprover` (SPEC §15) ties
Signal + Search + Archive + EvalSource together.

Run:
```
python chapters/ch02/spo_offline_loop.py
```

One invocation drives three rounds:

1. Reference pass: agent backed by the current live champion runs all
   eval questions.
2. SPO proposes one mutated candidate prompt.
3. Candidate pass: agent backed by the candidate runs the same questions.
4. `SwapAndAgree(PairwiseJudge)` compares answers question by question.
5. Aggregate verdict (mean score + majority vote) is recorded to the
   archive.

After three rounds, the archive contains three candidates with
measurements. **None of them are live.** The agent (run
`helixagent_v1.py` again) still serves the genesis prompt. This is the
**deploy gate**: offline improvement records candidates; promotion to
live is a separate, deliberate act.

Two things in the archive distinguish *best candidate* from *live
champion*:

- `archive.best()` returns the highest-scoring measured candidate.
- `archive.live_champion()` returns whatever was most recently promoted
  via `archive.promote()`. On first run, no promotion has happened, so
  this falls back to the genesis prompt.

Visualize what happened:

```
streamlit run helix/dashboard/app.py
```

The dashboard shows v1 as live, v2/v3/v4 as scored candidates, and a
"Promote v{N} → live" button on the best candidate. Click it (or call
`archive.promote()` from a notebook) and the next `helixagent_v1.py` run
serves the new version. That's the full self-improvement cycle.

---

## Cost expectations per SPO round

Rough order of magnitude on Haiku + Sonnet:

- Reference pass: 20 agent runs × ~5 model calls each — ~$0.10
- Candidate pass: same shape — ~$0.10
- Pairwise judge with `SwapAndAgree`: 20 × 2 judge calls on Sonnet — ~$0.30
- SPO proposer call: 1 × Sonnet — ~$0.01

Total per round: roughly **$0.50**. Three rounds: ~$1.50. Use Haiku for
the judge or skip `SwapAndAgree` for cheaper iterations during
development.

---

## What's deferred

- **Evolutionary search.** GEPA, multi-search strategy chains, tool
  description improvement. Chapter 3.
- **Memory tiers.** Episodic / semantic / procedural memory; agents that
  learn from past trajectories. Chapter 4.
- **Online improvement.** Auto-promotion, rolling spot-checks, no
  labeled eval set. Chapter 8 (pairs with HITL and live feedback).
- **Code-level artifacts.** Tool implementations, guardrails. Chapters
  11/12.

The single-improver, single-artifact, offline, human-gated pattern in
this chapter is the foundation everything else builds on.
