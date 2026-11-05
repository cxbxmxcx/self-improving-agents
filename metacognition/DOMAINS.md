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

Built in `examples/refunds` (`world.py`, `agent.py`, `study.py`, and the hand-written procedures in
`memories/`). It ran live twice on 2026-11-05: a first pass, then a second that applies the first pass's
lessons. The results are below the design.

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

**Results** (2026-11-05, `gpt-5.4-mini`, one sample per configuration: 1,083 live calls, $1.94). The recording
replays the whole study in about 15 seconds, and `tests/test_refunds.py` checks the numbers below.

| Returns solved | `none` | `low` | `medium` | `high` |
|---|---|---|---|---|
| No procedure (development returns) | 6/16 | 15/16 | 16/16 | 16/16 |
| Hand-written procedure (test returns) | 13/16 | 15/16 | 16/16 | 16/16 |

- **This domain needs little memory.** Without a procedure the agent clears the adequate bar at `low` and
  solves every return at `medium`, because the returns policy states its rules plainly. With the procedure the
  floors are the same and the cost barely moves ($0.0052 a run at `medium` with it, $0.0050 without). That's
  the study's thesis from the other side: memory pays when it carries knowledge the model can't derive, and
  here the model can.
- **A procedure's answer should speak its action tool's language.**
  - At `none` the hand-written procedure was right every time, yet the agent got 3 of 16 test returns wrong,
    and 2 of the 10 regular returns in each online arm.
  - The procedure said `{"action": "refund"}` and the agent called `issue_refund` with
    `method: "store_credit"`. The procedure's words (refund, credit) aren't the tool's (card, store_credit),
    and at `none` the agent sometimes translates by its own reading of the policy.
  - The expense procedure returns the amount the claim tool takes, and got 16 of 16 at `none`.
  - Design rule: have a procedure return the arguments its action tool takes.
  - The online arms also used a hand-set controller (`none` with a procedure), where D2's measured floors say
    `low` or `medium`.

| Online arms (60 returns; the policy changes at run 11) | checker | reviewers | judge |
|---|---|---|---|
| Wrong actions served | 13 | 19 | 11 |
| Dollars under-refunded (none were over-refunded) | $31.95 | $111.60 | $9.00 |
| First quarantine of a trusted procedure, at run | 12 | 22 | 8 |
| A relearned version trusted, at run | 20 | never (proposed at 57) | 21 |
| Learning spend | $0.40 | $0.10 | $0.33 |

- **Scarce, late reviews cost the most.**
  - With 30% of runs reviewed 10 runs late, the stale procedure stayed trusted through 11 holiday returns, and
    all of them were wrong.
  - It was quarantined only when reviews of the first ones arrived, at run 22.
  - Every one of those returns shorted the customer, $111.60 in all, three and a half times the checker's
    loss.
  - Learning then waited for four reviewed, accepted runs. A candidate came at run 57, and the stream ended
    before a person approved it.
  - Meanwhile the agent served at `high` without a procedure: mostly right, at the most expensive setting.
- **A judge on every run catches problems fast, with false alarms.**
  - It agreed with the audit on 56 of 60 runs.
  - One of its three false rejections fell on run 8, a correct regular return. With a real mistake at run 7,
    that quarantined the procedure before the policy changed. The false alarm happened to shield the holiday
    returns, which the agent then served at `high`.
  - Its one false acceptance let a wrong refund through.
  - Detection is fast but imprecise, so the design holds: a judge may raise the alarm, never install code
    alone.
- **A small replayed gate lets incomplete procedures through.**
  - In the checker and judge arms, learned versions passed the gate's 3 replayed cases, then failed within
    about ten holiday returns. The checker arm quarantined learned versions twice more, the judge arm once.
  - The gate's cases were the last 3 accepted runs: too few, and too alike, to show a version handles every
    holiday case. D1's gate used 6 trips held back because they hinge on the change, and its replacement
    held.
  - In the checker arm, the learning budget (4 attempts a day) stopped the retries from run 48, as designed.
- **Caveats:**
  - There is one sample per configuration.
  - The controller was set by hand, not from D2's floors.
  - The people are simulated: reviews are sampled at random, and approval comes automatically 5 runs after a
    proposal.

**Second pass** (2026-11-05, the same model: 765 live calls, $1.11; the no-procedure floors replayed free from
the first pass). `python -m examples.refunds.study --contract tool --controller measured --gate cases` applies
the first pass's three lessons at once:
- **The procedure answers in the action tools' terms:** `{"method": "card" | "store_credit" | "decline",
  "amount_usd"}`. `in_tool_terms` rewrites only the hand-written procedures' return lines, so the code is
  otherwise the same.
- **The arms take their reasoning settings from the measured floors:** `none` with the procedure and `low`
  without, the cheapest settings that solve 7 of 8.
- **The gate checks a candidate on the 6 held-out holiday returns, labeled.** It runs the procedure in the
  simulator and takes the audit as the label: `Gate(cases=..., practice=practice)`, the harness option this
  pass added. The learner still practices only by replaying accepted runs.

The new calls joined the first pass's recording, which replays both passes offline. `tests/test_refunds.py`
checks the numbers below.

| Returns solved with the hand-written procedure (test returns) | `none` | `low` | `medium` | `high` |
|---|---|---|---|---|
| Answering in its own words (first pass) | 13/16 | 15/16 | 16/16 | 16/16 |
| Answering in the tools' terms | 15/16 | 16/16 | 16/16 | 16/16 |

| Online arms: second pass (first pass) | checker | reviewers | judge |
|---|---|---|---|
| Wrong actions served | 3 (13) | 27 (19) | 4 (11) |
| Dollars over-refunded | $85.99 ($0) | $85.99 ($0) | $85.99 ($0) |
| Dollars under-refunded | $281.94 ($31.95) | $310.29 ($111.60) | $281.94 ($9.00) |
| First quarantine of a trusted procedure, at run | 11 (12) | 33 (22) | 11 (8) |
| A relearned version trusted, at run | 15 (20) | never: proposed at 56 (57) | 20 (21) |
| Learning spend | $0.17 ($0.40) | $0.17 ($0.10) | $0.13 ($0.33) |

- **Answering in the tools' terms ended the translation errors.**
  - With the procedure at `none`, the agent solved 15 of 16 test returns, up from 13.
  - In the online arms it got 1 of the 10 regular returns wrong, where the first pass got 2 wrong in each
    arm. That one was not a translation (see below).
- **The labeled gate admitted only complete versions, and they held.**
  - Each arm's learner wrote a version from 4 replayed runs, and each passed the 6 labeled returns.
  - All three are right on all 82 returns in the test, stream and gate splits: the checker's, the judge's,
    and the version the reviewers never approved.
  - No arm quarantined a relearned version. In the first pass, 3 replayed cases let incomplete versions
    through, and they were quarantined three more times.
- **The measured settings cost less and lost nothing.** Without a procedure, the agent got 86% of holiday
  returns right at the measured `low` (32 of 37) and at the first pass's hand-set `high` (43 of 50).
  - The floor was measured on regular returns. On holiday returns, neither setting clears the 7-of-8 bar.
- **What remains at `none` is the agent overruling a right procedure.**
  - Both of the checker arm's mistakes after the fix happened the same way. The agent guessed a customer id
    from the return number (C-209, C-713H) instead of taking it from the order. It then let what it read
    override the procedure.
  - At run 9 it refunded an opened beauty item the procedure had declined, $85.99 too much.
  - At run 23 it declined a $258.99 card refund because "the customer record was not found".
  - The prompt asks the agent to read everything and work the return out, which invites it to second-guess
    the procedure. The next rule to test: when a procedure is offered, the agent acts on its answer and
    doesn't redo its work.
- **With scarce, late reviews, detection waits for two reviewed failures.**
  - Among the first holiday returns the reviewers judged runs 12 and 23, so the second rejection arrived at
    run 33. By then 22 holiday returns had been served wrongly.
  - The first pass quarantined at run 22 only because a translation mistake at run 7 had been reviewed.
    That unrelated error gave the detector a head start, as the judge's false alarm did.
  - With the regular returns now right, that head start was gone. The detector's limit (2 rejections in its
    last 4 verdicts) suits verdicts on every run, and reacts slowly when 30% arrive, 10 runs late. A trusted
    person's rejection may deserve a lower limit; this is untested.
- **The judge was more accurate.** It agreed with the audit on 57 of 60 runs and accepted no wrong action.
  Its 3 false rejections, at runs 1, 14 and 46, never added up to a quarantine.
- **Dollar totals are lumpy.**
  - Two wrong actions account for $344.98 of the checker arm's errors.
  - A wrong decline costs a customer far more than a misapplied fee, so the count of wrong actions and the
    dollars can move in opposite directions.
  - The dollars also miss harm that isn't an amount: store credit where the card was due.
- **Caveats:**
  - There is one sample per configuration.
  - Identical requests share one recorded reply. The arms' first 10 runs are therefore one sample, and the
    run 9 and run 23 mistakes recur across arms.
  - The people are simulated.

**Follow-ups:**
- **Act on the procedure.** Re-measure the floors at `none` with a prompt that has the agent act on an
  offered procedure's answer, not redo its reads. About $0.10.
- **Detect sooner with scarce reviews.** Rerun the reviewers arm with a detector that quarantines on one
  rejection from a person. About $0.50.

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
