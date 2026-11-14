# A bank's support inbox

The production-scale test's third desk, the one we did not write. Its messages were written by real people and labeled by the datasets' own annotators. The routing policy that turns a topic into a ticket is ours, and it changes during the stream, as Dana's note does on the calendar desk.

## The data

Neither dataset is kept in this repository. Download them into `examples/scale/runs/outside_data/`, which git ignores.

| Dataset | Files | Source | License |
|---|---|---|---|
| Banking77 (Casanueva et al., 2020) | `banking77/train.csv`, `test.csv`, `categories.json` | https://github.com/PolyAI-LDN/task-specific-datasets, folder `banking_data` | CC BY 4.0 |
| CLINC150 (Larson et al., 2019) | `clinc150/data_full.json` | https://github.com/clinc/oos-eval, folder `data` | CC BY 3.0 |

Banking77's training split supplies the stream's messages, and its test split what is held back each day and the final exam. CLINC150 supplies two things: its out-of-scope queries, which must get no ticket, and the queries of four banking intents that mean exactly one of Banking77's topics (lost card, PIN change, exchange rate, declined card), which arrive from day 8 as messages from a different group of writers.

## The desk

| Part | What it is |
|---|---|
| `policy.py` | The 77 topics, their teams, what is urgent, who gets premium care, and the three versions of the policy |
| `world.py` | One message's world: the customer, the tools (`read_policy`, `read_customer`, `file_ticket`), the check, and the routing function's contract and reference |
| `stream.py` | Twenty days of 200 messages, the five changes, the sets held back each day, and the exam |
| `desk.py` | The agent, its loop and its decision tier, as settings of helix |
| `programs/` | The hand-written component: a router and a decision program that files a ticket with no language model |
| `scripted.py` | Scripted models for a soak that calls nothing |
| `run.py` | The driver: `start`, then `alone`, `loop`, `full` and `searches` |

A message is right when the ticket filed, or the absence of one, is what the policy in force asks for: the team that handles it and whether it is urgent. The topic named on the ticket is not judged, since two topics can share a team.
