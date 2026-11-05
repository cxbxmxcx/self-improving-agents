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
  - There is one sample per configuration; `--repeats 3` (about $2) would show the variance.
  - Everything ran on one model.
  - The policy change is one rule. The online loop's gate used the 6 held-back trips, so it never tested the
    replacement on a request it learned from.

## D2. Store refunds: can an agent learn safely when verdicts are scarce, late and human?

- **The agent:** a returns desk. It reads an order, the customer and the returns policy (read-only), then
  issues a refund or store credit and replies (acting).
- **The procedure:** `refund_decision(order_id, sku, reason) -> {"approve", "amount", "method"}`.
- **The policy's conventions can't be inferred from a request:**
  - return windows by category;
  - a restocking fee on opened electronics;
  - final-sale items;
  - store credit instead of a refund after the window;
  - a longer window for members;
  - a deduction for missing accessories.
- **Verdicts:** simulated reviewers judge a random 30% of runs, about ten runs later, through
  `meta.feedback(...)`. The simulator knows the right answer, but metacognition sees only the reviewers. A
  `Judge` with the policy as its rubric runs on every run. Its agreement with the reviewers is measured, and it
  doesn't drive the detector unless that's turned on.
- **Practice:** `ReplayPractice`, so learning never touches the simulator, only accepted runs' recorded worlds,
  with refunds dry-run.
- **Release:** installs wait for a person (approval "human"); the dashboard is the review queue.
- **Drift:** a holiday policy lengthens windows for November and December purchases, and waives the
  restocking fee.
- **Questions:**
  - With 30% of runs judged, late, how long does detection take, and what does that delay cost?
  - Does the judge shorten it without letting a bad version in?
  - Does a replay gate pass the same candidates an agent gate would?

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
