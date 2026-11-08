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

**Follow-ups** (2026-11-06: 404 live calls, $0.35, added to the same recording and checked by
`tests/test_refunds.py`):

- **Act on the procedure.** A third contract, `--contract act`, adds one sentence to the procedure's
  description: it reads the return, order, customer and policy itself, so call it first and act on its
  answer without reading them again. Measured at `none` on 76 returns (test and stream) with the
  hand-written procedure:

  | At `none`, 76 returns | Solved | Model calls a run | Dollars a run | Seconds a run |
  |---|---|---|---|---|
  | Tools' terms (second pass) | 74 | 4.17 | $0.0028 | 4.2 |
  | Tools' terms, and act on its answer | 76 | 3.0 | $0.0019 | 2.5 |

  - Told to act on the answer, the agent calls the procedure, acts and replies. It does no reads of its
    own, so it has nothing to overrule the procedure with. Both earlier failures (R209, H713) are gone,
    and runs are a third cheaper and 40% faster.
  - Every agent-side mistake in D2's later runs had this one cause. In the follow-up below, which still
    used the second pass's wording, the agent twice took a 15% restocking fee off a right holiday refund
    after the procedure had answered.
- **Detect sooner with scarce reviews.** `Detector(limits={"human": 1})` quarantines on one rejection from
  a person, while a judge's rejections still need two. The per-source limit is new in the harness. The
  reviewers arm, rerun with it (`--person-limit 1`):

  | Reviewers arm | Second pass | One rejection from a person |
  |---|---|---|
  | Wrong actions served | 27 | 15 |
  | Dollars under-refunded | $310.29 | $192.45 |
  | Quarantined at run | 33 | 22 |
  | A relearned version trusted, at run | never | 38 |

  - The first reviewed holiday failure (run 12, reviewed 10 runs late) now quarantines at run 22.
  - The learner wrote a version at run 33 from the one accepted, reviewed run it had. It passed the 6
    labeled returns, a person approved it at run 38, and it is right on all 82 returns.
  - Of the 15 wrong actions, 11 are the stale procedure serving before the first review arrived. With 30%
    of runs reviewed 10 runs late, the delay itself is now the cost. Two more are the agent overruling the
    relearned procedure (above).
  - The simulated reviewers are never wrong. With real people, one rejection should be enough only from a
    reviewer whose rejections you trust, and a judge keeps the higher limit.

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

**Built offline** on 2026-11-06 in `examples/invoices` (`world.py`, `agent.py`, `study.py`, `memories/`), with 15
tests in `tests/test_invoices.py`. It has not run live yet.
- **Clients:** Alder & Finch Clinics, Bramblewood Schools and Copperline Builders. Each client's readable terms
  say: pay for delivered quantities, tax its own categories, take the vendor's discount, and pay freight as
  invoiced.
- **Unwritten conventions** show only in each client's verdicts. Alder leaves freight out, Bramblewood rounds
  down to the dollar, and Copperline withholds 5% on materials. Three payments in four depend on one. This is
  knowledge the model can't derive, the side of the thesis D2 lacked.
- **Vendors and layouts:** Kestrel (key and value lines), Marlow (a ruled table), Tern & Daughters (a letter)
  and Quillon (semicolon rows).
- **Drift:** from stream request 21 Kestrel writes European dates and decimal commas. The parser written
  before the change misreads those documents without failing.
- **Procedures** answer in their action tools' terms, as D2 taught. `parse_invoice` returns `record_invoice`'s
  arguments and `amount_payable` returns `schedule_payment`'s. Vendor questions are in no procedure's class.
- **Arms:** `routed` (one shared Metacognition, calibrated routing, blame on what a run called), `unrouted`
  (both procedures offered to every request and both blamed) and `per_client` (`PerUser`). A relearned
  version must pass a labeled gate at a bar of 1.0.
- **What the scripted runs show** (mechanics only, not results):
  - Routing calibrated from labeled requests sends 16 of 16 test requests to their own class, against 0 of 16
    without thresholds.
  - In `routed`, only the parser is quarantined, and the labeled gate readmits it four runs later. In
    `unrouted`, the payable is quarantined with it and relearned for nothing.
  - In `per_client`, the shared parsing work is relearned three times over: 3 quarantines and 6 wrong
    intakes, against 1 and 2 when shared.
  - `discover` ranks the parsing work first.
- **Harness limits D3 found,** each with the workaround it uses:
  - `PerUser` scopes a whole Metacognition, so a parser shared by every client beside per-client payables
    can't be expressed. The study compares the two scopes instead.
  - One gate serves every procedure. Each practice passes the other procedure's cases, and gate totals count
    them, for example 24 of 24 when 6 cases are the parser's.
  - One learner serves every procedure. The study uses skill synthesis, which writes from all accepted runs
    (every layout) and repairs from the audit's reasons (the conventions).
  - Learning draws on the runs routed to a procedure. With calibrated routing that is right, but without
    routing each procedure also learns from the other's runs.
- **A live run** of every phase should cost about $1.60 to $1.90 on `gpt-5.4-mini` with the hand-written
  payable, and about $0.90 more to learn the payable from scratch (`--payable learn`).

**Live, partly** (2026-11-06, `gpt-5.4-mini`: 521 calls, $2.75; `tests/test_invoices.py` replays it):
- **The setup:** `--payable learn --selection --designed ...`. The payable starts untrusted, learner tournaments
  are on, and the two designs that transferred are approved as candidates.
- **It ran over the estimate.** It stopped at its $3 budget, 37 requests into the first arm (routed), before the
  unrouted and per-client arms ran. Learning with the default writer (`medium`) on long payment prompts cost
  most of it.

| Requests solved | `none` | `low` | `medium` | `high` |
|---|---|---|---|---|
| No procedure (development requests) | 4/16 | 8/16 | 9/16 | 8/16 |
| Hand-written procedures (test requests) | 16/16 | 16/16 | 16/16 | 16/16 |

- **Memory pays here, unlike in D2.** Without a procedure the agent never solves more than 2 of 8 payments: the
  clients' conventions are written nowhere. With the hand-written procedures it solves every request at `none`.
- **Routing** calibrated from 48 labeled requests sends 16 of 16 test requests to their own class.
- **`discover` ranks the payment work first and parsing second.** Payments spend the most reasoning and are
  accepted least often. The scripted test assumed parsing would come first.
- **In the online loop, a designed learner learned what no seed could.**
  - Skill synthesis, the arm's learner, wrote the payable twice, and the gate refused it twice: none of the
    18 labeled payments right.
  - At request 16 the tournament gave every seed and the second design none of the 18. The design searched
    on D1 and D2 got all 18.
  - The tournament chose it. Its version has served every payment right since.
- **The parser after Kestrel's layout change.** Skill synthesis came close to relearning it (20 and then 22 of
  24 gate cases), but the gate refused both. The run stopped at the next step, a parser tournament.
- **Next:** finish the routed arm (about $1 more), then the unrouted and per-client arms (about $2 to $3, less
  with a cheaper writer).

## D4. Model switch: do floors and learned procedures carry over between models?

- **Runs:** D1 and D3 on `claude-haiku-5-5` and `claude-sonnet-5-5` beside `gpt-5.4-mini` (`--model`,
  `--writer-model`, `--procedure-file`, `--payable-file`, and D3's `--contract act`).
- **Questions:**
  - What is each model's floor with and without a procedure?
  - Does a procedure written by one model's writer serve the other model's agent?
  - What does each model cost per accepted run?
- **Machinery:**
  - Claude models go through Anthropic's SDK (`helix.metacognition.claude`), behind the OpenAI-style call helix makes.
  - LiteLLM sends nothing for `none` on the Claude 5.5 family, which leaves thinking on. So `none` is
    `disabled` thinking on Haiku 5.5 and `between_tools` on Sonnet 5.5. Opus 5.5 has no `none` rung, because
    it can't turn thinking off.
  - Floors are kept per model, and the effort ladders send each model its own settings.

**Live** (2026-11-07, $6.66: Haiku $0.44, Sonnet $6.19, gpt-5.4-mini $0.15 for the act control;
`tests/test_d4.py` replays every run):

| D1 expenses, solved of 16 | `none` | `low` | `medium` | `high` | $ per request at `none` |
|---|---|---|---|---|---|
| gpt-5.4-mini, no procedure | 0 | 6 | 9 | 12 | $0.0021 |
| gpt-5.4-mini, learned procedure | 16 | 16 | 16 | 16 | $0.0022 |
| Haiku 5.5, no procedure | 12 | 15 | 16 | 15 | $0.0011 |
| Haiku 5.5, learned procedure | 16 | 16 | 16 | 16 | $0.0013 |
| Sonnet 5.5, no procedure | 15 | 16 | 16 | 16 | $0.0209 |
| Sonnet 5.5, learned procedure | 16 | 16 | 16 | 16 | $0.0237 |

| D3 invoices, solved of 16 | `none` | `low` | `medium` | `high` |
|---|---|---|---|---|
| Haiku 5.5, no procedure | 10 | 10 | 10 | 10 |
| Haiku 5.5, procedures in the tools' terms | 16 | 15 | 13 | 12 |
| Haiku 5.5, gpt-5.4-mini's designed payable | 16 | 11 | 11 | 10 |
| Haiku 5.5, told to act on the payable's answer | 16 | 16 | 16 | 16 |
| Sonnet 5.5, no procedure | 10 | 10 | 10 | 10 |
| Sonnet 5.5, procedures in the tools' terms | 10 | 11 | 10 | 9 |
| Sonnet 5.5, told to act on the payable's answer | 16 | 16 | not run | not run |
| gpt-5.4-mini, either contract | 16 | 16 | 16 | 16 |

- **Procedures carry over between models.**
  - Each model's writer learned D1's procedure from its own experience, and each version matched the
    hand-written reference on all 16 validation trips. gpt-5.4-mini's writer spent $0.43 (14 calls),
    Sonnet's $0.19 (16), and Haiku's $0.03 (16).
  - Any model's agent served any writer's procedure with the same results. The procedures gave the same
    answers, so the runs were the same calls, and they replayed from each other's recordings.
- **The more a model reasons, the more it overrules a procedure.**
  - With D3's hand-written procedures, Haiku went from 16 of 16 at `none` to 12 at `high`.
  - Sonnet, which keeps thinking between tool calls even at `none`, got at most 11 of 16.
  - It calls the payable, then works the amount out again and pays its own figure. In its words, the
    "tool looks like it rounds down". The rounding is one of the clients' unwritten conventions, the very
    thing the procedure learned.
  - D2's fix carries over. Once the payable's description says to call it first and pass its answer to
    `schedule_payment`, every model got every D3 request right at every setting run. Sonnet then stopped
    thinking (0 reasoning tokens) and cost 24% less than without a procedure, at 25% less latency.
- **A procedure pays least where the model needs it least.**
  - Each comparison is at the model's lowest setting with the most requests right, without and with the
    procedure.
  - gpt-5.4-mini: D1 cost 73% less and took 75% less time; D3 (told to act) 33% less on both.
  - Haiku: D1 cost 9% less and took 20% less time; D3 cost 27% less, with 6 more right.
  - Sonnet: D1 cost 19% more and took 54% longer with the procedure, at `none` as at any setting. It
    already solved D1, and its lowest setting still thinks.
- **What `none` means differs.**
  - Haiku with thinking off writes its working into the reply.
  - Sonnet's `between_tools` still thinks after tool results (453 reasoning tokens per D1 request at `none`).
  - Claude counts about twice gpt-5.4-mini's tokens for the same prompts, so dollars, not tokens, compare
    models.
- **Not run:** Sonnet's online loop and its D3 run with gpt-5.4-mini's designed payable; Opus 5.5.

## Learning about learning (A5): which learner, and can a search design a better one?

Built in `examples/learning`: `problems.py` builds learning problems from the domains' recordings, and
`study.py` runs the loop's learner tournament (A5, L2) and the design search (L3). The mechanism the search
designed is kept in `designed/`. It ran live on 2026-11-06 for $0.92 (108 live calls), and
`tests/test_learning.py` replays the whole study.

A learning problem holds the runs an agent made without a procedure, and the held-out cases the learned
function is scored on (does its answer get accepted?).

| Problem | Role | Runs to learn from (accepted) | Held-out cases |
|---|---|---|---|
| D1 expenses: the reimbursable total | training | 16 development trips at `low` (6) | 16 validation trips |
| D2 refunds: what a holiday return gets | training | the first 16 holiday returns at `low` (14) | the last 16 |
| Chapter 6's `multi_attendee` class | test | 16 development requests at `low` (4) | 16 validation requests |
| Chapter 6's `same_time` class | test | 16 development requests at `low` (2) | 16 validation requests |

Each learner works under a regime that sets the writer's effort and allows 12 writer calls a problem. Chapter
7's `low_writer` gives the writer `low`. `weak_writer` gives it `none`, the cheapest setting chapter 7 lets a
design choose.

**L2, learner tournaments** (held-out cases accepted, of 16):

| Seed | D1, `low_writer` | D2, `low_writer` | D1, `weak_writer` | D2, `weak_writer` |
|---|---|---|---|---|
| value learning | 16 | 0 (ran out of writer calls) | 16 | 0 (ran out) |
| reflection | 16 (cheapest) | 0 | 16 (cheapest) | 0 |
| success distillation | 16 | 16 (cheapest) | 0 | 8 (best) |
| transition distillation | 16 | 16 | 16 | 0 |
| skill synthesis | 16 | 16 | 16 | 0 |

- **No learner is best everywhere, so choosing one per procedure pays.** The tournament picked reflection for
  D1 and success distillation for D2.
- **The default learner failed D2.** Value learning writes one candidate per accepted run, and D2's 14 runs
  overran a 12-call budget. Reflection learned D1 in one call but never learned D2.
- **With a weak writer, D2 defeats every seed.** The best scored 8 of 16.

**L3, a design search** under `weak_writer`, where no seed learns both training problems: the best average 0.5.
- GEPA started from transition distillation, the cheapest of the seeds tied at 0.5.
- Three rewrites took it to 0.59, 0.75 and then 1.00: the design learns both D1 and D2 perfectly with the
  writer at `none`.
- New spend was $0.44, against a $6 cap.
- **What the design does** (`designed/designed_ee25cd476662.py`, written by the search):
  - It drafts from the accepted and the rejected runs, as transition distillation does.
  - It checks the draft on every run it learned from, as skill synthesis does.
  - It repairs from those failures at most three times, the last time from scratch.
  - It keeps whichever candidate the practice accepted most, not the last one written.
  - Its prompts target the failures the search's feedback showed: invented API names, and refusals the
    policy doesn't call for.
- **Chapter 6's calendar classes couldn't tell it apart from the seed.** No design saw them. It scored 2 of 16
  and 1 of 16, 0.09, the same as transition distillation. With no reasoning, no learner learns those classes'
  unwritten conventions from 4 or 2 accepted runs. The gate's rule (don't lose on the test problems) accepted
  it on that tie.

**Follow-ups** (2026-11-06, $1.18 in all, replayed by `tests/test_learning.py` and `tests/test_refunds.py`):

- **A test that discriminates: D3, and the design transfers.** D3's intake and payment requests became learning
  problems (`--test invoices_intake invoices_payment`). Every seed and the design learned each one with the
  writer at `none` (`--phases transfer`), held-out cases accepted:

  | Learner | D3 intake (of 12) | D3 payment (of 16) |
  |---|---|---|
  | value learning, reflection, success distillation, transition distillation | 0 | 0 |
  | skill synthesis | 2 | 0 |
  | the design, searched on D1 and D2 only | 7 | 12 |

  - D3 is a domain the design never saw, and on it the design learned more than every seed.
  - On payments it learned 12 of 16 from 3 accepted runs, by checking itself and repairing from the audit's
    reasons, which name each client's unwritten convention as it breaks.
- **The design competes in D2's online loop** (`--writer-effort none --learner reflection --selection
  --designed ...`, the checker arm):
  - Reflection failed the gate twice, 0 of 6 both times, at runs 15 and 19.
  - At run 23 the tournament scored value learning, skill synthesis and the design each 6 of 6 on the gate's
    labeled returns.
  - It chose skill synthesis as the cheapest ($0.027 against the design's $0.039). That version was trusted
    at run 23, and the arm served 4 wrong actions.
  - On 6 gate cases the design ties the best seeds, and cost decides. It wins where the seeds fail, as on D3.
- **Repeated searches vary a great deal.** These are two more searches from other random starting points,
  each with fresh designer samples. A label in the designer's prompt keeps the recording from replaying one
  search's replies into the next.

  | Search (random starting point) | Best on training | On D3 (intake, payment) |
  |---|---|---|
  | 7, the first | 1.00 | 0.67 (7/12, 12/16) |
  | 11 | 0.75 | 0.21 (5/12, 0/16) |
  | 23 | 0.91 | 0.00 (0/12, 0/16) |

  - Every search beat the best seed's 0.50 on training, but only the first transferred well.
  - The search from 23 did second best on training and transferred nothing, so training score doesn't predict
    transfer.
  - The gate's verdict now says when a design only ties on the test problems: "no sign it transfers".
- **The worker designs a learner when none can learn a procedure.** `LearnerSelection(design=DesignPolicy())`
  starts a design search in the worker, never in a serving process. It fires when a tournament's winner still
  fails the gate.
  - The search trains on the procedure's own learning problem, and optionally tests on problems from other task
    classes.
  - An accepted design is registered as proposed, and logged and published (`mechanism_proposed`). A person
    approves it before it competes.
  - This is scripted in `tests/test_a5.py` and hasn't run live.
- **Lessons:**
  - Choose a learner per procedure, by measurement. No seed was best everywhere, and the default failed D2
    outright.
  - A search can design a learner that makes up for a cheaper writer and transfers to a domain it never saw.
    The gain came from checking against its own runs and keeping its best attempt.
  - It doesn't happen every time. One search in three found such a design, and training scores didn't show
    which. Judge designs on test problems where the seeds score low but above zero, and prefer a design that
    beats the seed there over one that ties.
  - A tournament on a few gate cases breaks ties by cost. That is right for serving, but it can hide a stronger
    learner until the cases are harder.
- **Caveats:**
  - There is one sample of each search and tournament.
  - Scores are acceptance on held-out cases, not chapter 7's match against a reference.
  - The worker's own design search is tested only with scripted models.
- **D3 live (see D3 above): the design wins a tournament in a real loop.**
  - It learned D3's payable when the default writer, every seed and the second design could not: 18 of 18
    labeled payments, against none.
- **A stricter gate.** A design is now accepted only if it beats the best seed on the test problems.
  - A tie is no sign of transfer: it is refused unless `ties=True`, and then ranks below a design that
    transfers.
  - Each design keeps a standing (`transfers`, `untested`, `ties`), and listings and tournaments order designs
    by it.
  - Two earlier verdicts changed. The first design is refused on the calendar classes, where it only tied,
    though it transfers to D3. The repeat from random starting point 23 is refused too, and was dropped from
    `designed/`.
- **Validation inside the search** (`--validation invoices_intake --test invoices_payment`, $0.36):
  - Every candidate was also scored on D3 intake while the search ran (GEPA's Pareto front of training and
    validation; the designer saw training only), and D3 payment stayed held out.
  - The pick learned training at 0.91 and intake at 5 of 12, where the seed learns none.
  - On payments it tied the seed at zero, so the gate refused it.
  - Transfer to one unseen class didn't carry to another. Choosing on validation helps pick, but the verdict
    belongs to a class the search never touched.
- **Next:**
  - Finish D3's online arms.
  - Search under the default writer, where D3 shows the design matters most.
  - Repeat each search more times, to put a rate on transfer.

## Replacing the LLM: decision programs (new chapter 7 direction)

Chapter 6 removed the reasoning; the agent still read the request, called the procedure and acted. A decision
program removes the agent too: code does the routing, the procedure and the action, and Jev (TypeSafe's System
One model, $0.042 per million input tokens, output free) answers typed judgments. When unsure it hands off. A
learning mechanism writes the program from the LLM's judged runs. (`helix/metacognition/decide.py`, `examples/decisions/`,
`tests/test_decide.py`; live 2026-11-08, $1.51.)

- **The split.** D1's and D3's requests are templated, so `examples/decisions/wordings.py` asks for the same
  work in other words: casual, shorthand ("invoice 201"), a distractor id, and buried in a longer note. There are
  also out-of-scope requests, and two sets of near misses that use the job's own words to ask for something else.
- **The mechanisms** (`meta/programs/`): `program_synthesis` writes a program from the runs and repairs it from
  practice. `program_judgment` does the same but tells the writer that what a request asks for is Jev's call,
  and it practices on near misses too.

Held-out varied wordings, with the LLM (gpt-5.4-mini, procedures at `none`) taking every hand-off:

| | D1 right of 68 | D3 right of 62 | $ per request | Seconds |
|---|---|---|---|---|
| LLM alone | 49 | 48 | $0.0023 to $0.0026 | 2.4 |
| Hand-written regex + Jev | 50 | 48 | $0.0012 to $0.0015 | 1.3 to 1.4 |
| Learned, keywords for intent | 65 | 62 | $0.00015 to $0.00019 | 0.17 |
| Learned, Jev judges intent | 66 | 62 | $0.00019 to $0.00022 | 0.21 to 0.23 |

Near misses in the held-out set, of 20 per domain, acted on when they should have been handed off:

| | D1 | D3 |
|---|---|---|
| LLM alone | 9 | 6 |
| Hand-written regex + Jev | 0 | 0 |
| Learned, keywords for intent | 20 | 16 |
| Learned, Jev judges intent | 0 | 2 |

- **What was learned.** The programs worked out how ids are really written (`T-201`, `INV-201`). The LLM got
  every shorthand request wrong, because nothing tells it that format. The learned programs got them all right.
- **Without being told, the writer reached for word lists.** Those programs scored well on the first test and
  failed the near misses. Telling the mechanism that intent is a judgment for Jev fixed it, and the writer then
  wrote the Jev questions itself. The first near-miss set informed that change, so the second was written before
  any such program existed.
- **Limits.** Only two domains, and both have a code checker and clean ids. A program's hand-off goes to the LLM
  here, which still acts on many near misses, so a hand-off of "this isn't the job" should go to a person or a
  read-only agent. The judgment programs are single attempts.
- **Next:** a search over these mechanisms, and the triage domain, where the work itself is a judgment.
