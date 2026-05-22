# Chapter 2: Prompts that improve themselves

Files in this chapter:

- **`helixagent_v0.py`** — the baseline: RAG agent with a fixed system prompt,
  one retrieval tool, working memory only. No improvement loop.
- **`helixagent_v1.py`** — same agent, but the system prompt comes from the
  persistent SQLite Archive. On first run the archive is seeded with v0's
  genesis prompt; subsequent runs read whatever SPO has accepted as champion.
- **`spo_offline_loop.py`** — the **offline** improvement driver. One
  invocation runs one SPO round: reference pass over the eval questions,
  SPO mutation, candidate pass, pairwise-judge each question with
  `SwapAndAgree`, aggregate verdicts (mean score for ordering + majority
  vote for accept/reject), record to archive. Candidates are not served
  until a human promotes one via `archive.promote()` (or the dashboard
  button). Re-run to do more rounds.
- **`spo_online_loop.py`** — the **online** counterpart. Simulates live
  traffic by replaying eval questions one at a time, spot-checks responses
  with `LiveTrajectoryJudge` (an absolute rubric judge — no reference
  answer required), and triggers SPO when the rolling average drops. If
  the candidate beats the reference on a small shadow sample, the script
  publishes `CandidateWins(mode="online", auto_promote=True)` and the
  default promotion hook calls `archive.promote()` automatically. The
  next request reads the new live champion.
- **`eval_questions.json`** — 20 hand-curated questions across 4 difficulty
  bands (factual / disambiguation / multi-hop / trap). Reference answers
  derived from the arXiv corpus. Drives the eval harness and SPO judge.
- **`eval_harness.py`** — runs an agent against the eval set; writes one JSONL
  record per question to `runs/<label>_<timestamp>.jsonl`. Optional `judge`
  callback scores each record (used by v1 / SPO).

## Prerequisites

1. Build the corpus once:
   ```
   python ingestion/build_index.py
   ```
2. Set an LLM provider key. LiteLLM picks up the env var for whichever model
   you choose (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

## Run v0

```
python chapters/ch02/helixagent_v0.py
```

Edit the `question` variable in `main()` to experiment with a different
prompt. The agent and tools are exposed as `build_agent()` and `ask_one()` so
you can also import them from a REPL or a notebook.

## Run the eval harness

```
python chapters/ch02/eval_harness.py
```

Runs HelixAgent v0 against all 20 questions in `eval_questions.json` and
writes one JSONL line per question to `runs/v0_<timestamp>.jsonl`. Each line
holds the question, the reference answer, the agent's answer, the full
trajectory, and basic metrics (latency, tool-call count, model-call count).
At the end the script prints per-band summary statistics.

No judging happens at v0; the `judgment` field stays null. v1 plugs a pairwise
judge into `run_eval(judge=...)` to score answers.

## Run v1 offline (one SPO round)

```
python chapters/ch02/spo_offline_loop.py
```

One invocation runs one round:

1. Reference pass: agent backed by current-best prompt runs all eval questions.
2. SPO proposes one mutated candidate prompt.
3. Candidate pass: agent backed by the candidate runs the same questions.
4. Pairwise judge (with `SwapAndAgree`) compares answers question by question.
5. Aggregate verdict (mean score + majority vote) is recorded to the archive.

The candidate becomes the new top-scoring "best candidate" automatically if
its mean score is the highest in the archive (because `archive.best()` orders
by score). Note that "best candidate" is not the same as the **live champion**
the running agent serves: SPO here runs in offline mode, so the candidate
sits in the archive until a human reviews and promotes it via
`archive.promote()`. Re-run `spo_offline_loop.py` to do another round; each
round picks up the current best candidate as the reference. DESIGN_NOTES.md
section 10.

Per-round summaries are appended to `runs/spo_rounds.jsonl`.

After SPO has run and you have promoted a winning candidate (either through
the dashboard's "Promote v{N} → live" button or programmatically via
`archive.promote()`), `python chapters/ch02/helixagent_v1.py` and
`python chapters/ch02/eval_harness.py` (with v1's factory) automatically use
the live champion prompt. Until then, the agent serves the genesis prompt.

## Run v1 online (auto-promoting, no labels)

```
python chapters/ch02/spo_online_loop.py
```

The online pattern requires no labeled eval set at runtime. Live traffic is
simulated by replaying eval questions one at a time. Each response is
scored 0-1 by `LiveTrajectoryJudge` — an absolute rubric judge that reads
the question, the agent's tool calls, and the final answer, and rates
grounding, correctness, abstention, and concision. Spot-checks accumulate
in a rolling window. When the rolling average drops below the threshold,
SPO proposes a candidate, the candidate is shadow-evaluated on the next
few requests, and if it beats the reference, the script fires a
`CandidateWins` event that the default promotion hook auto-promotes.

The layer rule is enforced structurally: `ImproverPolicy(mode=ONLINE)`
refuses L3 (planner / monitor) and L4 (code) artifacts at construction.
Prompts are L1, so this example is safe.

The online example differs from offline along three axes:

| Axis | Offline | Online |
|------|---------|--------|
| trigger | manual / scheduled | rolling score drop |
| signal | pairwise judge over labeled pairs | absolute rubric judge |
| promotion | human via dashboard | auto via `CandidateWins` hook |
| safety | any artifact layer | L1/L2 only (enforced) |

Cost shape: every spot-checked request invokes the agent plus the judge,
plus the SPO proposer when a candidate is triggered, plus the shadow-eval
runs (one extra agent run per side, per shadow question). Tune
`SAMPLE_RATE`, `ROLLING_WINDOW`, and `SHADOW_SAMPLE` at the top of the
script to budget the example.

## Visualize what happened

After any improvement run, launch the platform dashboard:

```
streamlit run helix/dashboard/app.py
```

It reads from `chapters/ch02/runs/helix_archive.sqlite` by default. Pages
walk you through the run: the current champion, the artifact lineage tree
(colored by Search method), side-by-side prompt diffs, per-question judge
verdicts, trajectory replays with cross-prompt comparison, and a round-by-
round score timeline.

The dashboard is platform-level (lives in `helix/dashboard/`), so the same
tool works for any chapter's runs.

## Cost expectations per SPO round

Rough order of magnitude on gpt-4o-mini + gpt-4o:

- Reference pass: 20 agent runs × ~5 model calls + tool calls each — ~$0.10
- Candidate pass: same shape — ~$0.10
- Pairwise judge with `SwapAndAgree`: 20 × 2 judge calls on gpt-4o — ~$0.30
- SPO proposer call: 1 × gpt-4o — ~$0.01

Total per round: roughly **$0.50**. Three rounds: ~$1.50. Use a smaller judge
or skip `SwapAndAgree` for faster, cheaper iterations during development.

## What v0 is, in spec terms

Composition (Spec §11.1):
- One **Artifact** (the system prompt, kind=prompt).
- One **Tool** (`retrieve`, calling LanceDB hybrid search).
- **Working memory** only; no episodic, semantic, or procedural memory.
- The fixed **Agent loop** (Spec §6.1) with hook points wired but empty.
- A **Trajectory** is recorded on every run.

What is deliberately missing:
- No Signal, no Search, no Archive. Those arrive in v1.
- No episodic memory write-back. That's Ch 3.
- No reflection. That's Ch 5.

## What v1 adds (preview)

- The system prompt becomes the seed Artifact of an SPO search.
- A pairwise LLM-as-Judge plays the role of Signal.
- An Archive (SQLite-backed) records each variant with its measurement.
- The next agent run reads the current best variant from the archive instead
  of the hand-authored genesis prompt.

This is the entire self-improvement loop at minimal complexity.
