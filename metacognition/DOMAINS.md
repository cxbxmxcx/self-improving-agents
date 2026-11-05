# Example domains: exploring metacognition by building real agents

Each domain is built with the harness the way a deployed agent would be (tools, a procedure, a verdict
source, calibration, deployment, then drift), and each asks one question about metacognition. The plan is in
[ARCHITECTURE.md](ARCHITECTURE.md) §6. This file holds each domain's protocol and, once run, its results.
Live runs are recorded, so every result here replays without an API key.

## D1. Expense claims: does the study's result hold on a second domain?

**Protocol.** `python -m examples.expenses.study --mode record --budget 4 --output examples/expenses/runs/d1`
runs four phases on `gpt-5.4-mini`, 16 trips per split:

1. **floors:** the development trips at all four settings, without a procedure.
2. **learn:** value learning with the domain-neutral prompts, from the medium and high development runs,
   checked on the validation trips against the hand-written procedure.
3. **memory:** the test trips at all four settings with the learned procedure.
4. **online:** 6 test trips, then 14 after the policy change (receipts convert at their own month's rates).
   The gate holds back 6 trips from after the change.

Calls are recorded to `examples/expenses/recorded/study.jsonl`.

**Results** (2026-11-05, `gpt-5.4-mini`, one sample per configuration: 456 live calls, $1.05). The recording
replays the whole study in about 15 seconds, and `tests/test_expenses.py` checks every number below.

| Trips solved (cost per run) | `none` | `low` | `medium` | `high` |
|---|---|---|---|---|
| No procedure (development trips) | 0/16 ($0.0021) | 6/16 ($0.0040) | 9/16 ($0.0057) | 12/16 ($0.0081) |
| Learned procedure (test trips) | 16/16 ($0.0022) | 16/16 ($0.0027) | 16/16 ($0.0031) | 16/16 ($0.0038) |

- **Without a procedure, reasoning helps but isn't enough.** The agent goes from 0 to 12 of 16 as reasoning
  rises, and never reaches the adequate bar of 14, so there is no floor. The calendar study's equivalent
  was 2, 4, 12 and 13 of 16.
- **The domain-neutral prompts learned a correct procedure.** Value learning took 32 development runs (21
  accepted) and 14 writer calls ($0.43). Its function was accepted on all 16 validation trips and returned
  exactly what the hand-written reference returns on all 16. The study learned with the book's calendar
  prompts, so this is the first live evidence for the generic ones.
- **With the procedure, the floor is `none`.** All 16 test trips are solved at every setting. At `none` a
  run costs $0.0022, 27% of the $0.0081 the agent spends at `high` without it, where it still solves only 12.
- **Through the policy change, the study's timeline repeats almost exactly:**
  - The stale procedure's first two claims after the change were rejected for converting at the trip's
    starting month, and the detector quarantined it after run 8.
  - Four runs went to `high` without it, and three were accepted.
  - At run 12, value learning wrote a replacement from those four runs (3 writer calls, $0.05). It passed
    the gate on all 6 held-back trips and was installed.
  - All 8 remaining requests ran at `none` and were accepted.
  - Learning paid for itself after about 8.5 requests; the calendar's figure was 7.6.
- **Caveats:**
  - Everything ran on one model.
  - The policy change is one rule. The online loop's gate used the 6 held-back trips, so it never tested the
    replacement on a request it learned from.

**Repeats** (2026-11-05: 791 more live calls, $0.95, so $2.00 for D1 in all). Both measurement phases ran three
times over with the learned procedure, from
`--phases floors memory --repeats 3`. The first pass replays the run above, and each later pass gets its own
replies.

| Solved in each repeat (of 16) | `none` | `low` | `medium` | `high` |
|---|---|---|---|---|
| No procedure | 0, 0, 0 | 6, 5, 4 | 9, 10, 10 | 12, 12, 11 |
| Learned procedure | 16, 16, 16 | 16, 16, 16 | 16, 16, 16 | 16, 16, 16 |

- **The no-procedure numbers hold.** The totals are 0, 15, 29 and 35 of 48, and no setting reaches the
  adequate bar.
- **Without the procedure, the agent is inconsistent as well as inaccurate.** At `high`, only 8 of the 16 trips
  were solved all three times. Three were solved once and one never, so the same request passes one day and
  fails the next.
- **With the procedure, every trip passed every time at every setting**, 192 of 192 runs. A procedure lowers
  the cost and removes the variance.

The recording replays the repeats offline in about 30 seconds, and the D1 test checks them.

## D2. Store refunds: can an agent learn safely when verdicts are scarce, late and human?

Built in `examples/refunds` (`world.py`, `agent.py`, `study.py`, and the hand-written
`memories/refund_decision.py`), with no live run yet.

- **The agent:** the returns desk at a store. It reads the return, its order, the customer and the returns
  policy (all read-only), then acts once: `issue_refund` (to the card, or as store credit) or
  `decline_return`.
- **The procedure:** `refund_decision(return_id) -> {"action": "refund" | "credit" | "decline",
  "amount_usd"}`. Procedures can't import, so the hand-written one does its own date arithmetic.
- **Conventions that live only in the policy:**
  - windows count from delivery and vary by category, and members get 15 more days;
  - up to 30 days after the window, a return gets store credit instead of a refund;
  - opened electronics carry a 15% restocking fee, and opened beauty items can't come back;
  - missing accessories are deducted;
  - shipping is never refunded, and final-sale items can't be returned;
  - an item that arrived damaged gets everything back, whatever else applies.
- **Returns:** each regular split mixes refunds, credit and declines, and together the splits cover every
  convention. The audit names the rule a wrong action most likely broke.
- **Drift:** a holiday policy. Items delivered from November 15 through December 31 can be returned to the
  card until January 31, with no restocking fee. Every return in the stream after the change hinges on it.
- **Arms:** `python -m examples.refunds.study --phases online`. Each arm runs 10 regular returns, then 50
  under the holiday policy, starting from the hand-written procedure. They differ only in who judges:
  - **checker:** the audit judges every run as it ends, and a passing version installs itself.
  - **reviewers:** people judge 30% of runs, 10 runs late, through `meta.feedback`, and a person approves
    proposals 5 runs after the gate passes them.
  - **judge:** the reviewers, plus a `Judge` on every run whose verdicts reach the detector.
- **Learning:** value learning with the domain-neutral prompts. Practice is `ReplayPractice`: a candidate
  is tried only in accepted runs' recorded worlds, with refunds dry-run, never in the simulator.
- **Measures:** every run is also audited, whoever judged it. The study reports:
  - wrong actions served, and the dollars over- and under-refunded;
  - the run at which the stale procedure was quarantined, relearned and trusted again;
  - the cost of learning;
  - how often the judge agreed with the audit.
- **Questions:**
  - With 30% of runs judged, late, how long does detection take, and what does the delay cost?
  - Does the judge shorten it without letting a bad version in?
  - Can replayed practice gate a candidate with only the reviewed runs to go on?

## D3. Invoice intake: how does metacognition behave with several procedures and several customers?

- **The agent:** accounts payable for several client companies. It reads an invoice (raw text, in each
  vendor's own layout), the purchase order and the client's payment terms, then posts a ledger entry.
- **The procedures:** `parse_invoice(doc_id)` (fields from the vendor's layout) and
  `amount_payable(invoice_id)` (each client's discount, tax and partial-delivery rules). Payable amounts
  differ by client, so `amount_payable` is learned per client with `PerUser`.
- **Traffic:** a mix of the two kinds of work and vendor questions that belong to neither, so routing is
  calibrated from labeled requests.
- **Drift:** one vendor changes its invoice layout.
- **Questions:**
  - Does routing keep each request with its own procedure, with the others running at the agent's setting?
  - Does blame quarantine only the parser while the payable procedure stays trusted?
  - Do per-client procedures learn from their own clients' runs?
  - Does `metacog discover` find the parsing work before anyone declares it?

## D4. Model switch: do floors and learned procedures carry over between models?

- **Runs:** D1's floors and memory phases, and D3's, on `claude-haiku-5-5` beside `gpt-5.4-mini`. The keys
  for both providers are in `.env`.
- **Questions:**
  - What is each model's floor with and without a procedure?
  - Does a procedure written by one model's writer serve the other model's agent?
  - What does each model cost per accepted run?
- **Machinery:** floors are kept per model, so both sets sit in one store, and the effort ladders send each
  provider its own reasoning settings.
