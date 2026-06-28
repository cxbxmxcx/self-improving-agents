# The search-method cost ladder (Chapter 3 findings)

This is the demonstration of when a more expensive search is worth its cost. The three search methods of this chapter, SPO, GEPA, and DGM, are run as the same improvement loop on one task, swapping only the Search. The result is a clean progression where each more elaborate method reaches a strictly higher score, and the reason it does so is mechanical and inspectable rather than asserted.

## The result

| method | best score | relative cost | where it stops, and why |
| --- | --- | --- | --- |
| SPO | 0.63 | lowest | one blind mutation per round on outcome-only feedback; it cannot localize the fix and plateaus flat |
| GEPA | 0.86 | medium | reflects on the agent's execution trace to diagnose omitted rules, then the population converges and loses diversity |
| DGM | 1.00 | highest | an accumulating quality-diversity archive explores the branches GEPA discarded and recovers the last rule |

Every run uses the same mid-tier model (Sonnet) for the agent, the proposer, and the reflector. Nothing about model capability changes between the three. The gap is produced entirely by the search mechanism, which is the honest form of the claim.

## The task

The agent answers a vague analytics question over a small simulated company database, chaining query tools to produce a leaderboard. The catch is a set of business rules the request leaves implicit: revenue is net of discounts and a refund column, orders that are cancelled, returned, internal, or unpaid do not count, "last quarter" is relative to a fixed today rather than the latest data, and totals are grouped by rep. A vague prompt gets several of these wrong, every tool call still succeeds, and the reported number is silently off.

Scoring is plain Python with no LLM judge and no personas. A canonical pipeline computes the true leaderboard, and the agent's answer is graded against it. The smooth scorer gives partial credit per rule, so applying more rules moves the score up as a gradient rather than an all-or-nothing jump, which is what lets a plateau be visible. The task and scorer live in `agents/revenue.py`.

## Why the search is hard

A capable proposer one-shots rules it can infer from domain knowledge, which flattens any gap. The difficulty here comes from interference, not obscurity: there are enough independent rules that no single mutation reliably holds them all, so fixing some tends to drop others. With a mid-tier proposer and outcome-only feedback, that interference is enough to trap greedy hill-climbing in a local optimum well below the ceiling.

This is why the configuration matters. A frontier proposer removes the plateau by solving everything at once, and a too-weak proposer cannot drive the reflection and crossover the elaborate methods rely on. The mid-tier proposer is the band where the methods separate.

## The mechanisms

SPO mutates the current best once per round and keeps the result if a pairwise judge prefers it. Its only signal is the outcome, "the totals are too high," which says nothing about which rule is missing. The proposer makes generic edits that never name the omitted filters, and the score sits flat at 0.63 for ten rounds.

GEPA adds reflection over a population. Its reflector reads the agent's actual tool calls, sees that the agent filtered status but never filtered channel or payment and never subtracted refunds, and lists those as the fixes. That diagnosis is far richer than the outcome alone, so the mutation targets the real cause and the score climbs to 0.86. The small population then converges, every candidate makes the same omission, and GEPA loses the diversity it would need to find the last rule.

DGM keeps an archive that never discards a candidate and samples parents by quality and diversity rather than only the current best. It revisits lower-scoring, structurally different candidates that GEPA would have thrown away, and on one of those branches a variant phrases the status filter as "status equals ok," which excludes the returned orders the whole GEPA population missed. That candidate scores 1.0, and because the archive retained it, DGM finishes at the ceiling.

## The lesson

The cost ladder is real but conditional. SPO is the right default when the feedback already localizes the fix or the landscape is smooth, because it is the cheapest by a wide margin. GEPA earns its extra cost when the execution trace contains information the outcome does not, which is common for multi-step tool use. DGM earns its further cost when the landscape is deceptive enough that a converging population gets stuck, and its diversity-preserving archive is what escapes that.

## Reproducing

Run the three loops in order; each prints its climb. They share the task and scorer in `agents/revenue.py` and differ only in the Search.

- `python chapters/ch03/04_revenue_spo_loop.py` (SPO, plateaus at 0.63)
- `python chapters/ch04/03_revenue_gepa_loop.py` (GEPA, reaches 0.86)
- `python chapters/ch04/06_revenue_dgm_loop.py` (DGM, reaches 1.00)

`chapters/ch03/02_revenue_check.py` validates the substrate before any search run: genesis low, oracle 1.0, fully deterministic at temperature 0.
