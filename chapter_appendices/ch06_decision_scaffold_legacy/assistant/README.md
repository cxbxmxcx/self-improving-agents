# The chapter 6 assistant

The assistant works for Maya Rivera, and for Leo Chen on two memory requests, in a local week of calendars and an inbox. Nothing here touches a real account: every tool acts on an in-memory copy of [world.json](world.json), and `snapshot`, `restore`, and `fork` let evaluation and shadow runs start from the same week.

## The world

The week runs Monday 2026-10-19 to Friday 2026-10-23. The contacts are chosen so that each metacognitive move has something to do. Two people are named Sam, and Maya has two 1:1s, two reviews on Wednesday, and two meetings on Thursday.

Several events belong to other people: moving Priya's planning meeting or Sam Lee's design review changes their calendars. The inbox holds six messages, three of which ask two questions at once.

## Tools and the monitor

[tools.py](tools.py) binds eight tools to one environment: `read_calendar`, `book_time`, `move_event`, `read_inbox`, `read_message`, `draft_reply`, `send_reply`, and `review`. `send_reply` always changes something for another person, and `move_event` does when someone else organized the event. The monitor watches both, and `AssistantEnvironment.consequential` is the safety truth the evaluation checks against.

`book_time` checks only the user's own calendar, the way a real calendar sends an invitation over someone's busy time. That is the fast path's documented failure on constrained requests, and checking everyone's calendar is what the planner adds.

## The requests

[requests.json](requests.json) holds 40 training requests, 20 held-out requests, and one walkthrough request, in five categories; [requests.md](requests.md) shows them as tables, and [requests_view.py](requests_view.py) rewrites it after an edit. Each request carries hidden truth, which [environment.py](environment.py) checks: a booking's constraints, an event's new time, the points an answer or a reply must make, or the options a clarifying question must name.

A reply or a move that is prepared correctly and held for confirmation counts as accepted, because nothing changed without the user. Memory requests ship with the episodes or lessons they rely on, which the evaluation preloads with `preload_memory`.

## The scripted stand-in

[scripted.py](scripted.py) answers the agent's model calls from each request's script, so the listings run without keys. It reads only what a model would see: the request, the system message, which shows the path and what memory recalled, and the tool results. It never reads the hidden truth, and its scripted outcomes are tested against the checker on every path.

The stand-in's mistakes are written down, not discovered. On constrained requests its fast path checks only Maya's calendar, and on ambiguous ones it guesses. These runs show the mechanics; measured performance comes from live runs.

## Evaluation

[evaluation.py](evaluation.py) holds what listings 6.3 and 6.4 share. Every run starts from the same week with only its request's prepared memory, and nothing is written back, so conditions differ only in what the meta level decides. It defines the conditions, builds the paired runs, and computes calibration and the precision and recall of each intervention.

`WEIGHTS` sets what a dollar and a second are worth in outcome points, which is a judgment the comparison depends on. A cent costs 0.1 points, and so do ten seconds of waiting.

## Recordings

[recording.py](recording.py) decides where decisions and search proposals come from. Replay reads the recording and fails loudly on a miss; record mode sends only what the recording lacks to the live model and keeps the answer. Each question therefore has one recorded answer, even though Jev can answer a repeated question slightly differently.

[recorded/](recorded/) holds Jev's answers (`decisions.jsonl`), the LLM decision model's (`llm_decisions.jsonl`), the search proposer's replies (`proposals.jsonl`), and the configuration the recorded search kept (`searched.json`). All were recorded on 2026-10-12 with the stand-in; refresh them with `--record`, in the order 6.1, 6.2, 6.4, 6.3.
