"""Turn the calendar desk's three runs into measures, and say which of the nine pass criteria each meets.

    python -m examples.scale.desk_report                 the paid run's results
    python -m examples.scale.desk_report --soak          a scripted soak's results

The criteria and their thresholds are the ones SCALE_TEST.md fixed before any run. What this file adds is the
arithmetic behind each, written down so it is the same for a soak and for the paid run.

    common requests   the requests two assistants both served: every fifth for the loop, every 25th for all three
    serving cost      what a request's own model and decision-model calls cost
    learning cost     everything else the part's recordings hold: builder calls, and the agent's runs at night
    whole stream      an assistant that served a share of the stream is scaled to all of it: its serving cost
                      times the stream's length over the requests it served, plus its learning cost once, since
                      learning follows changes
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
BUILDER = "claude"  # a model whose name holds this wrote code; every other recorded model is the agent's
WITHIN = 200  # requests a change may take to recover in


def load(folder: Path, name: str) -> dict[str, Any] | None:
    path = folder / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def wilson(right: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% interval for a share, which stays inside 0 and 1 for small counts."""
    if not total:
        return 0.0, 0.0
    share = right / total
    middle = (share + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, middle - half), min(1.0, middle + half)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))] if ordered else 0.0


def stream_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted((row for row in run["rows"] if row["phase"] == "stream"), key=lambda row: row["seq"])


def wrong_action(row: dict[str, Any]) -> bool:
    """A request that was answered wrongly by doing something: a booking the checker rejected."""
    return not row["right"] and row["booked"] > 0


def served(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What a set of requests cost, took and got right."""
    total = len(rows)
    right = sum(row["right"] for row in rows)
    low, high = wilson(right, total)
    seconds = [row["seconds"] for row in rows]
    return {"requests": total, "right": right, "share_right": right / total if total else 0.0,
            "share_right_low": low, "share_right_high": high,
            "dollars_per_request": statistics.fmean(row["dollars"] for row in rows) if rows else 0.0,
            "seconds_mean": statistics.fmean(seconds) if rows else 0.0,
            "seconds_median": statistics.median(seconds) if rows else 0.0, "seconds_p95": percentile(seconds, 0.95),
            "wrong_actions": sum(wrong_action(row) for row in rows),
            "by_programs": sum(row["outcome"] in ("program", "wrong") for row in rows),
            "wrong_by_programs": sum(row["outcome"] == "wrong" for row in rows)}


def costs(run: dict[str, Any]) -> dict[str, float]:
    """The whole stream's cost for one assistant, split into serving and everything else. An assistant that served a
    share of the stream has its serving cost scaled to the stream's length."""
    rows = stream_rows(run)
    length = run["settings"]["per_day"] * run["settings"]["days"]
    scale = length / len(rows) if rows else 0.0
    serving = sum(row["dollars"] for row in rows)
    by_model = (run.get("spent") or {}).get("by_model")
    if by_model is not None:
        # The recordings hold every call the part made: its serving, its exam and repeats, and its learning. What
        # the served requests cost comes off, and the rest is learning: builder calls, and the agent's and the
        # decision model's work at night.
        recorded = sum(cell["dollars"] for cell in by_model.values())
        learning = max(0.0, recorded - sum(row["dollars"] for row in run["rows"]))
        builder = sum(cell["dollars"] for model, cell in by_model.items() if BUILDER in model)
    else:  # a soak has no recording: the learning log's own dollars stand in
        learning = builder = sum(float(entry.get("dollars") or 0.0) for entry in run.get("learning", []))
    return {"serving": serving, "learning": learning, "builder": builder, "whole_stream": serving * scale + learning,
            "serving_whole_stream": serving * scale, "requests_whole_stream": length}


def recovery(rows: list[dict[str, Any]], first: int, window: int = 40, level_over: int = 200,
             slack: float = 0.03) -> dict[str, Any]:
    """Requests from a change until the share right over `window` requests is back within `slack` of its level
    over the `level_over` requests before the change. None if it never is."""
    before = rows[max(0, first - level_over):first]
    level = sum(row["right"] for row in before) / len(before) if before else 1.0
    found = None
    for end in range(first + window, len(rows) + 1):
        if sum(row["right"] for row in rows[end - window:end]) / window >= level - slack:
            found = end - first
            break
    return {"level_before": level, "requests_until_back": found,
            "wrong_in_first_200": sum(not row["right"] for row in rows[first:first + 200])}


def consistency(run: dict[str, Any]) -> dict[str, Any]:
    """Of the exam's requests, how many got the same verdict in all three servings."""
    verdicts: dict[str, list[bool]] = {}
    for row in run["rows"]:
        if row["phase"] in ("exam", "repeat2", "repeat3"):
            verdicts.setdefault(row["id"], []).append(row["right"])
    full = {key: value for key, value in verdicts.items() if len(value) == 3}
    same = sum(len(set(value)) == 1 for value in full.values())
    return {"requests": len(full), "same_all_three": same, "share": same / len(full) if full else 0.0,
            "right_all_three": sum(all(value) for value in full.values())}


def payback(full: dict[str, Any], other_per_request: float) -> int | None:
    """The request at which the full system's running cost, learning included, falls below another assistant's
    and stays there. Learning is charged on the day's last request it followed."""
    rows = stream_rows(full)
    learning = costs(full)["learning"]
    per_request = learning / len(rows) if rows else 0.0  # spread evenly: nights are not priced one by one
    spent = ahead = 0.0
    crossing = None
    for n, row in enumerate(rows, 1):
        spent += row["dollars"] + per_request
        ahead = other_per_request * n - spent
        if ahead > 0 and crossing is None:
            crossing = n
        elif ahead <= 0:
            crossing = None
    return crossing


def report(folder: Path) -> dict[str, Any]:
    full, loop, alone, searches = (load(folder, name) for name in ("full", "loop", "alone", "searches"))
    runs = {"full": full, "loop": loop, "alone": alone, "searches": searches}
    out: dict[str, Any] = {"criteria": [], "arms": {}, "changes": [],
                           "missing": [name for name, run in runs.items() if run is None]}
    if full is None:
        return out
    rows = stream_rows(full)
    out["arms"]["full"] = {**served(rows), **costs(full), "stopped": full["stopped"]}
    out["exam"] = {"full": served([row for row in full["rows"] if row["phase"] == "exam"])}

    def criterion(number: int, name: str, passed: bool | None, measured: str) -> None:
        out["criteria"].append({"number": number, "criterion": name, "passed": passed, "measured": measured})

    # ---- against the agent with Chapter 6's loop, on the requests both served
    if loop is not None:
        loop_rows = stream_rows(loop)
        shared = {row["seq"] for row in loop_rows}
        mine, theirs = served([row for row in rows if row["seq"] in shared]), served(loop_rows)
        loop_costs, full_costs = costs(loop), costs(full)
        out["arms"]["loop"] = {**theirs, **loop_costs, "stopped": loop["stopped"]}
        out["common_with_loop"] = {"full": mine, "loop": theirs}
        out["exam"]["loop"] = served([row for row in loop["rows"] if row["phase"] == "exam"])
        criterion(1, "Cheaper to serve", mine["dollars_per_request"] <= 0.70 * theirs["dollars_per_request"],
                  f"${mine['dollars_per_request']:.5f} a request against ${theirs['dollars_per_request']:.5f}, "
                  f"{mine['dollars_per_request'] / max(theirs['dollars_per_request'], 1e-12):.0%} of the baseline")
        criterion(2, "Pays for its learning", full_costs["whole_stream"] < loop_costs["whole_stream"],
                  f"${full_costs['whole_stream']:.2f} for the stream with ${full_costs['learning']:.2f} of learning, "
                  f"against ${loop_costs['whole_stream']:.2f} with ${loop_costs['learning']:.2f}")
        criterion(3, "Faster", mine["seconds_mean"] <= 0.70 * theirs["seconds_mean"]
                  and mine["seconds_p95"] <= 1.10 * theirs["seconds_p95"],
                  f"mean {mine['seconds_mean']:.2f} s against {theirs['seconds_mean']:.2f} s, 95th percentile "
                  f"{mine['seconds_p95']:.2f} s against {theirs['seconds_p95']:.2f} s")
        criterion(4, "As accurate", mine["share_right"] >= theirs["share_right"] - 0.02,
                  f"{mine['right']}/{mine['requests']} right against {theirs['right']}/{theirs['requests']}")
        program_rate = out["arms"]["full"]["wrong_by_programs"] / max(1, out["arms"]["full"]["by_programs"])
        criterion(5, "Safer", program_rate <= 0.005 and mine["wrong_actions"] <= theirs["wrong_actions"],
                  f"programs acted wrongly on {out['arms']['full']['wrong_by_programs']} of "
                  f"{out['arms']['full']['by_programs']} requests they served ({program_rate:.2%}); wrong actions "
                  f"{mine['wrong_actions']} against {theirs['wrong_actions']} on the common requests")
        out["payback_against_loop"] = payback(full, theirs["dollars_per_request"] + loop_costs["learning"]
                                              / max(1, loop_costs["requests_whole_stream"]))
    # ---- after every change
    from examples.scale.desk_stream import plan

    for day in plan(full["settings"]["days"]):
        if day.change:
            first = next((n for n, row in enumerate(rows) if row["day"] == day.number), None)
            if first is not None:
                out["changes"].append({"day": day.number, "change": day.change, **recovery(rows, first)})
    if out["changes"]:
        slowest = [c["requests_until_back"] for c in out["changes"]]
        criterion(6, "Robust", all(n is not None and n <= WITHIN for n in slowest),
                  "requests until back to its earlier level after each change: "
                  + ", ".join("never" if n is None else str(n) for n in slowest))
    # ---- the same request served three times
    if loop is not None and any(row["phase"] == "repeat3" for row in full["rows"]):
        mine, theirs = consistency(full), consistency(loop)
        out["consistency"] = {"full": mine, "loop": theirs}
        criterion(7, "More predictable", mine["share"] >= theirs["share"],
                  f"{mine['same_all_three']}/{mine['requests']} requests with one verdict in three servings against "
                  f"{theirs['same_all_three']}/{theirs['requests']}")
    # ---- five searches from the same start
    if searches is not None and searches["searches"]:
        floor = searches["floor"]["right"]
        admitted = [s for s in searches["searches"] if s["admitted"]]
        good = [s for s in admitted if s["held_out_right"] >= floor]
        out["searches"] = searches["searches"]
        criterion(8, "Repeatable", len(admitted) >= 4 and len(good) == len(admitted),
                  f"{len(admitted)} of {len(searches['searches'])} searches admitted; held-out right "
                  f"{[s['held_out_right'] for s in admitted]} against {floor} for the agent at its floor")
    # ---- against the agent alone at high reasoning
    if alone is not None:
        alone_rows = stream_rows(alone)
        shared = {row["seq"] for row in alone_rows}
        mine, theirs = served([row for row in rows if row["seq"] in shared]), served(alone_rows)
        alone_costs, full_costs = costs(alone), costs(full)
        out["arms"]["alone"] = {**theirs, **alone_costs, "stopped": alone["stopped"]}
        out["common_with_alone"] = {"full": mine, "alone": theirs}
        out["worth_it"] = {"cost": full_costs["whole_stream"] <= 0.25 * alone_costs["whole_stream"],
                           "accuracy": mine["share_right"] >= theirs["share_right"]}
        criterion(9, "Worth it against a first deployment", all(out["worth_it"].values()),
                  f"${full_costs['whole_stream']:.2f} for the stream against ${alone_costs['whole_stream']:.2f}, "
                  f"{full_costs['whole_stream'] / max(alone_costs['whole_stream'], 1e-12):.0%}; "
                  f"{mine['right']}/{mine['requests']} right against {theirs['right']}/{theirs['requests']}")
        out["payback_against_alone"] = payback(full, theirs["dollars_per_request"])
    # ---- the verdict
    passed = {c["number"]: c["passed"] for c in out["criteria"]}
    failed = {number for number, ok in passed.items() if not ok}
    # In part: accuracy, safety and recovery hold, and what fails is a cost or a time. Criterion 9 has a cost half
    # and an accuracy half, and only its cost half may fail.
    only_cost_or_time = failed <= {1, 2, 3, 9} and (9 not in failed or out.get("worth_it", {}).get("accuracy"))
    if len(passed) < 9:
        out["verdict"] = f"incomplete: {len(passed)} of 9 criteria measured"
    elif not failed:
        out["verdict"] = "holds"
    elif all(passed[n] for n in (4, 5, 6)) and only_cost_or_time:
        out["verdict"] = "holds in part"
    else:
        out["verdict"] = "does not hold"
    # ---- by day, for the figures
    out["days"] = []
    for number in sorted({row["day"] for row in rows}):
        today = [row for row in rows if row["day"] == number]
        cell = served(today)
        out["days"].append({"day": number, **{k: cell[k] for k in ("requests", "right", "dollars_per_request",
                                                                   "seconds_mean", "by_programs", "wrong_by_programs",
                                                                   "wrong_actions")},
                            "read_only": sum(row["outcome"] == "read_only" for row in today),
                            "by_agent": sum(row["outcome"] in ("llm", "llm_wrong", "lost") for row in today)})
    out["nights"] = full.get("nights", [])
    kept = ("name", "mechanism", "version", "episodes", "writer_calls", "dollars", "outcome", "at_ord")
    out["learning"] = [{key: entry.get(key) for key in kept} for entry in full.get("learning", [])]
    return out


def show(out: dict[str, Any]) -> None:
    for name, arm in out["arms"].items():
        print(f"{name:<6} {arm['right']}/{arm['requests']} right ({arm['share_right']:.1%}), "
              f"${arm['dollars_per_request']:.5f} and {arm['seconds_mean']:.2f} s a request, "
              f"${arm['whole_stream']:.2f} for the whole stream (${arm['learning']:.2f} learning)"
              + (f"  [{arm['stopped']}]" if arm["stopped"] else ""))
    for change in out["changes"]:
        back = change["requests_until_back"]
        print(f"   day {change['day']:>2}, {change['change']}: back after "
              f"{'never' if back is None else back} requests, {change['wrong_in_first_200']} wrong in the first 200")
    for c in out["criteria"]:
        print(f"{c['number']}. {c['criterion']:<36} {'pass' if c['passed'] else 'FAIL'}  {c['measured']}")
    print(f"Verdict: {out.get('verdict')}" + (f"; missing {out['missing']}" if out["missing"] else ""))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--soak", action="store_true")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    folder = args.results or HERE / "runs" / "desk_calendar" / ("soak" if args.soak else "results")
    out = report(folder)
    show(out)
    target = args.output or folder / "report.json"
    target.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
