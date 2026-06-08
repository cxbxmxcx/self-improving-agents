"""A deterministic data-investigation task for the Chapter 3 improvement loop.

The agent answers a vague analytics question ("top reps last quarter") by chaining
query tools over a tiny in-memory company database. The catch is several business
rules the request leaves implicit: revenue is net of discounts AND a refund column,
"last quarter" is relative to a fixed TODAY (not the latest data present), orders
with status 'cancelled' or 'returned' do not count, and totals are grouped by sales
rep. A vague genesis prompt gets some wrong; every tool call still "succeeds" and
the number is silently off. The improvement loop searches the system prompt to pin
the rules, and a plain-code scorer grades the answer against a canonical pipeline.

No LLM judge and no personas: the score is computed by re-running the canonical
query in Python and comparing the agent's reported leaderboard to the ground truth.
The four rules are near-orthogonal, so partial credit is separable, which is what
gives the population (GEPA) and archive (DGM) searches room to beat single-mutation
hill-climbing (SPO).
"""
from __future__ import annotations

import contextvars
from datetime import date
from typing import Any

from helix.agent import Agent
from helix.artifact import Artifact, Subtype, genesis
from helix.tools import tool

SYSTEM_PROMPT_ID = "prompt.revenue.system"
DEFAULT_MODEL = "claude-sonnet-4-5"

# "Today" is fixed so "last quarter" is deterministic: 2026-04-15 is in 2026-Q2,
# so last quarter is 2026-Q1 (Jan-Mar). The April orders are the "latest quarter
# in the data" trap a naive agent falls into.
TODAY = date(2026, 4, 15)

REPS = [
    {"rep_id": "R1", "rep_name": "Ada"},
    {"rep_id": "R2", "rep_name": "Ben"},
    {"rep_id": "R3", "rep_name": "Cy"},
    {"rep_id": "R4", "rep_name": "Dee"},
]
CUSTOMERS = [
    {"customer_id": "C1", "customer_name": "Acme"},
    {"customer_id": "C2", "customer_name": "Beta"},
    {"customer_id": "C3", "customer_name": "Gamma"},
    {"customer_id": "C4", "customer_name": "Delta"},
]
PRODUCTS = [
    {"product_id": "P1", "product_name": "Widget"},
    {"product_id": "P2", "product_name": "Gadget"},
    {"product_id": "P3", "product_name": "Gizmo"},
]
# Gotchas a capable agent misses by default: orders with status 'cancelled' OR
# 'returned' must be excluded (it tends to exclude only cancelled), and April
# orders are next quarter, not last. O4 (Dee, cancelled) and O12 (Ben, returned)
# are the status traps; their line totals are small, so including them shifts a
# top rep's value without reordering the board, keeping partial credit smooth.
ORDERS = [
    {"order_id": "O1", "customer_id": "C1", "rep_id": "R1", "order_date": "2026-01-10", "status": "ok", "channel": "direct", "payment": "paid"},
    {"order_id": "O2", "customer_id": "C2", "rep_id": "R2", "order_date": "2026-01-20", "status": "ok", "channel": "direct", "payment": "paid"},
    {"order_id": "O3", "customer_id": "C1", "rep_id": "R3", "order_date": "2026-02-05", "status": "ok", "channel": "direct", "payment": "paid"},
    {"order_id": "O4", "customer_id": "C3", "rep_id": "R4", "order_date": "2026-02-15", "status": "cancelled", "channel": "direct", "payment": "paid"},
    {"order_id": "O5", "customer_id": "C2", "rep_id": "R4", "order_date": "2026-03-01", "status": "ok", "channel": "direct", "payment": "paid"},
    {"order_id": "O6", "customer_id": "C4", "rep_id": "R2", "order_date": "2026-03-10", "status": "ok", "channel": "direct", "payment": "paid"},
    {"order_id": "O7", "customer_id": "C3", "rep_id": "R3", "order_date": "2026-03-20", "status": "ok", "channel": "direct", "payment": "paid"},
    {"order_id": "O8", "customer_id": "C1", "rep_id": "R4", "order_date": "2026-03-25", "status": "ok", "channel": "direct", "payment": "paid"},
    {"order_id": "O12", "customer_id": "C2", "rep_id": "R2", "order_date": "2026-02-20", "status": "returned", "channel": "direct", "payment": "paid"},
    # Two more gotchas: an internal-channel order (inter-company transfer, not real
    # revenue) and an unpaid (pending) order. A naive agent counts both.
    {"order_id": "O20", "customer_id": "C1", "rep_id": "R3", "order_date": "2026-02-12", "status": "ok", "channel": "internal", "payment": "paid"},
    {"order_id": "O21", "customer_id": "C4", "rep_id": "R2", "order_date": "2026-03-15", "status": "ok", "channel": "direct", "payment": "pending"},
    {"order_id": "O9", "customer_id": "C2", "rep_id": "R3", "order_date": "2026-04-05", "status": "ok", "channel": "direct", "payment": "paid"},
    {"order_id": "O11", "customer_id": "C4", "rep_id": "R1", "order_date": "2025-11-15", "status": "ok", "channel": "direct", "payment": "paid"},
]
# A 'refund' column on line items reduces net revenue; a capable agent applies the
# discount but ignores refunds unless told. O7 (Cy) carries the refund trap.
LINE_ITEMS = [
    {"order_id": "O1", "product_id": "P1", "qty": 2, "unit_price": 100, "discount": 0.0, "refund": 0},
    {"order_id": "O2", "product_id": "P2", "qty": 2, "unit_price": 50, "discount": 0.0, "refund": 0},
    {"order_id": "O3", "product_id": "P3", "qty": 1, "unit_price": 120, "discount": 0.1, "refund": 0},
    {"order_id": "O4", "product_id": "P2", "qty": 1, "unit_price": 40, "discount": 0.0, "refund": 0},
    {"order_id": "O5", "product_id": "P1", "qty": 1, "unit_price": 100, "discount": 0.0, "refund": 0},
    {"order_id": "O6", "product_id": "P3", "qty": 4, "unit_price": 120, "discount": 0.8, "refund": 0},
    {"order_id": "O6", "product_id": "P1", "qty": 1, "unit_price": 100, "discount": 0.0, "refund": 0},
    {"order_id": "O7", "product_id": "P2", "qty": 6, "unit_price": 50, "discount": 0.0, "refund": 50},
    {"order_id": "O8", "product_id": "P3", "qty": 1, "unit_price": 120, "discount": 0.0, "refund": 0},
    {"order_id": "O12", "product_id": "P1", "qty": 1, "unit_price": 50, "discount": 0.0, "refund": 0},
    {"order_id": "O20", "product_id": "P1", "qty": 3, "unit_price": 50, "discount": 0.0, "refund": 0},
    {"order_id": "O21", "product_id": "P2", "qty": 2, "unit_price": 50, "discount": 0.0, "refund": 0},
    {"order_id": "O9", "product_id": "P1", "qty": 5, "unit_price": 100, "discount": 0.0, "refund": 0},
    {"order_id": "O11", "product_id": "P1", "qty": 3, "unit_price": 100, "discount": 0.0, "refund": 0},
]

TABLES: dict[str, list[dict]] = {
    "reps": REPS, "customers": CUSTOMERS, "products": PRODUCTS,
    "orders": ORDERS, "line_items": LINE_ITEMS,
}


# ---------------------------------------------------------------------------
# Ground truth and scorer (plain code; this is what the agent must reproduce)
# ---------------------------------------------------------------------------

def _last_quarter_label(today: date) -> str:
    q = (today.month - 1) // 3 + 1
    y = today.year
    q -= 1
    if q == 0:
        q, y = 4, y - 1
    return f"{y}-Q{q}"


def _quarter_of(iso_date: str) -> str:
    y, m = int(iso_date[:4]), int(iso_date[5:7])
    return f"{y}-Q{(m - 1) // 3 + 1}"


def ground_truth(top_n: int = 3) -> list[tuple[str, float]]:
    """The canonical pipeline, in plain Python: net revenue by rep for last
    quarter, excluding cancelled orders, top N. This defines the correct answer."""
    target = _last_quarter_label(TODAY)
    name_of = {r["rep_id"]: r["rep_name"] for r in REPS}
    excluded = {"cancelled", "returned"}
    rep_of = {
        o["order_id"]: o["rep_id"]
        for o in ORDERS
        if o["status"] not in excluded
        and o.get("channel", "direct") == "direct"   # exclude internal transfers
        and o.get("payment", "paid") == "paid"        # exclude unpaid (pending) orders
        and _quarter_of(o["order_date"]) == target
    }
    net: dict[str, float] = {}
    for li in LINE_ITEMS:
        rid = rep_of.get(li["order_id"])
        if rid is None:
            continue
        line = li["qty"] * li["unit_price"] * (1 - li["discount"]) - li.get("refund", 0)
        net[rid] = net.get(rid, 0.0) + line
    ranked = sorted(((name_of[r], round(v, 2)) for r, v in net.items()), key=lambda x: -x[1])
    return ranked[:top_n]


def _parse_answer(answer: Any) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for item in answer or []:
        if isinstance(item, dict):
            name = item.get("rep_name") or item.get("name") or next(iter(item.values()), None)
            val = item.get("revenue") if "revenue" in item else item.get("value")
            if val is None:
                nums = [v for v in item.values() if isinstance(v, (int, float))]
                val = nums[0] if nums else None
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            name, val = item[0], item[1]
        else:
            continue
        try:
            out.append((str(name).strip(), float(val)))
        except (TypeError, ValueError):
            continue
    return out


def score_revenue(answer: Any, top_n: int = 3) -> float:
    """Fraction of the top-N leaderboard slots the agent got right, where a slot is
    right only if the rep name and the revenue (to the cent) match the ground truth
    in the correct rank position. Graded, so getting some rules right earns credit."""
    truth = ground_truth(top_n)
    got = _parse_answer(answer)
    hits = 0
    for i, (tn, tv) in enumerate(truth):
        if i < len(got) and got[i][0].lower() == tn.lower() and abs(got[i][1] - tv) <= 0.01:
            hits += 1
    return hits / len(truth) if truth else 0.0


def score_smooth(answer: Any, top_n: int = 3) -> float:
    """Smooth score: for each rep that belongs in the true top-N, credit how close
    its reported revenue is to the truth (1 when exact, decaying with relative
    error, 0 if the rep is missing). Unlike the exact-match score this gives partial
    credit for partial rule-application, so improvement shows as a gradient rather
    than an all-or-nothing jump."""
    truth = ground_truth(top_n)
    got = {str(n).strip().lower(): v for n, v in _parse_answer(answer)}
    credits = []
    for tn, tv in truth:
        pv = got.get(tn.lower())
        if pv is None:
            credits.append(0.0)
        else:
            credits.append(max(0.0, 1.0 - abs(pv - tv) / max(abs(tv), 1.0)))
    return sum(credits) / len(credits) if credits else 0.0


def score_smooth_with_feedback(answer: Any, top_n: int = 3) -> tuple[float, str]:
    """Smooth score plus per-rep feedback: which reps are off and in which direction,
    without the correct numbers, so the search must discover the rules."""
    truth = ground_truth(top_n)
    got = {str(n).strip().lower(): v for n, v in _parse_answer(answer)}
    notes = []
    for tn, tv in truth:
        pv = got.get(tn.lower())
        if pv is None:
            notes.append(f"{tn}: not reported")
        elif abs(pv - tv) > 0.01:
            notes.append(f"{tn}: total is " + ("too high" if pv > tv else "too low"))
    return score_smooth(answer, top_n), ("all reps correct" if not notes else "; ".join(notes))


def score_with_feedback(answer: Any, top_n: int = 3) -> tuple[float, str]:
    """Score plus coarse feedback for the proposer: which leaderboard positions are
    wrong and in which direction, but NOT the correct numbers, so the search has to
    discover the rule rather than memorize the answer."""
    truth = ground_truth(top_n)
    got = _parse_answer(answer)
    notes = []
    for i, (tn, tv) in enumerate(truth):
        if i >= len(got):
            notes.append(f"position {i + 1}: no rep reported")
        elif got[i][0].lower() != tn.lower():
            notes.append(f"position {i + 1}: wrong rep '{got[i][0]}'")
        elif abs(got[i][1] - tv) > 0.01:
            notes.append(f"position {i + 1}: '{tn}' total is " + ("too high" if got[i][1] > tv else "too low"))
    feedback = "all positions correct" if not notes else "; ".join(notes)
    return score_revenue(answer, top_n), feedback


# ---------------------------------------------------------------------------
# Per-run query session (a contextvar, so concurrent rollouts auto-isolate)
# ---------------------------------------------------------------------------

_SESSION: contextvars.ContextVar = contextvars.ContextVar("revenue_session", default=None)


def _sess() -> dict:
    s = _SESSION.get()
    if s is None:
        s = {"tabs": {}, "n": 0}
        _SESSION.set(s)
    return s


def _store(rows: list[dict]) -> dict:
    s = _sess()
    s["n"] += 1
    h = f"t{s['n']}"
    s["tabs"][h] = rows
    cols = list(rows[0].keys()) if rows else []
    return {"handle": h, "columns": cols, "row_count": len(rows), "sample": rows[:2]}


def _rows(handle: str) -> list[dict]:
    return _sess()["tabs"].get(handle, [])


def make_revenue_tools() -> list[Any]:
    """The eight chained query tools. Each returns a handle to its result rows that
    the next tool consumes, so answering forces a real tool chain."""

    async def load_table(name: str) -> dict:
        if name not in TABLES:
            return {"error": f"no table '{name}'. available: {list(TABLES)}"}
        return _store([dict(r) for r in TABLES[name]])

    async def filter_rows(handle: str, column: str, op: str, value: Any) -> dict:
        def keep(r: dict) -> bool:
            x = r.get(column)
            if op == "==":
                return x == value
            if op == "!=":
                return x != value
            if op == ">=":
                return x >= value
            if op == "<=":
                return x <= value
            if op == ">":
                return x > value
            if op == "<":
                return x < value
            if op == "between":
                return value[0] <= x <= value[1]
            if op == "in":
                return x in value
            return False

        return _store([r for r in _rows(handle) if keep(r)])

    async def join_tables(left: str, right: str, left_key: str, right_key: str) -> dict:
        index: dict = {}
        for r in _rows(right):
            index.setdefault(r.get(right_key), []).append(r)
        out = []
        for lr in _rows(left):
            for rr in index.get(lr.get(left_key), []):
                merged = dict(lr)
                for k, v in rr.items():
                    if k not in merged:  # keep the left value on key collisions
                        merged[k] = v
                out.append(merged)
        return _store(out)

    async def compute_column(handle: str, name: str, expr: str) -> dict:
        # Arithmetic over existing columns, e.g. "qty * unit_price * (1 - discount)".
        # builtins are stripped so the expression can only touch the row's columns.
        out = []
        for r in _rows(handle):
            try:
                r2 = dict(r)
                r2[name] = eval(expr, {"__builtins__": {}}, r)  # noqa: S307 (teaching sim, no builtins)
                out.append(r2)
            except Exception as exc:  # surface to the model
                return {"error": f"bad expr '{expr}': {exc}"}
        return _store(out)

    async def date_bucket(handle: str, date_column: str) -> dict:
        out = []
        for r in _rows(handle):
            r2 = dict(r)
            try:
                r2["quarter"] = _quarter_of(str(r.get(date_column, "")))
            except Exception:
                r2["quarter"] = ""
            out.append(r2)
        return _store(out)

    async def aggregate(handle: str, group_by: str, agg: str, column: str) -> dict:
        groups: dict = {}
        for r in _rows(handle):
            groups.setdefault(r.get(group_by), []).append(r)
        out = []
        for g, rs in groups.items():
            vals = [float(x.get(column, 0) or 0) for x in rs]
            if agg == "sum":
                v = sum(vals)
            elif agg == "count":
                v = float(len(rs))
            elif agg == "avg":
                v = sum(vals) / len(vals) if vals else 0.0
            elif agg == "max":
                v = max(vals) if vals else 0.0
            elif agg == "min":
                v = min(vals) if vals else 0.0
            else:
                return {"error": f"bad agg '{agg}'. use sum|count|avg|max|min"}
            out.append({group_by: g, column: round(v, 2)})
        return _store(out)

    async def sort_top_n(handle: str, column: str, n: int, descending: bool = True) -> dict:
        rows = sorted(_rows(handle), key=lambda r: float(r.get(column, 0) or 0), reverse=descending)
        return _store(rows[:n])

    async def summarize(handle: str, name_column: str, value_column: str) -> dict:
        """Emit the final leaderboard the scorer reads, as [[name, value], ...]."""
        answer = [[r.get(name_column), round(float(r.get(value_column, 0) or 0), 2)] for r in _rows(handle)]
        return {"answer": answer}

    descriptions = {
        load_table: "Load a named table (reps, customers, products, orders, line_items). Returns a handle plus its columns, row count, and a 2-row sample.",
        filter_rows: "Keep rows matching a condition: filter_rows(handle, column, op, value). op is one of ==, !=, >=, <=, >, <, between, in. Returns a new handle.",
        join_tables: "Inner-join two handles on keys: join_tables(left, right, left_key, right_key). Returns a new handle with both sides' columns.",
        compute_column: "Add a column from an arithmetic expression over existing columns: compute_column(handle, name, expr), e.g. expr='qty * unit_price * (1 - discount)'.",
        date_bucket: "Add a 'quarter' column (like '2026-Q1') derived from a date column: date_bucket(handle, date_column).",
        aggregate: "Group and aggregate: aggregate(handle, group_by, agg, column). agg is sum|count|avg|max|min. Returns one row per group.",
        sort_top_n: "Sort by a column and keep the top n: sort_top_n(handle, column, n, descending=True).",
        summarize: "Report the final answer as a leaderboard: summarize(handle, name_column, value_column). Call this last.",
    }
    return [tool(description=desc)(fn) for fn, desc in descriptions.items()]


def reconstruct_answer(trajectory: Any) -> list:
    """Read the agent's final leaderboard from the last summarize() result."""
    pending = None
    answer: list = []
    for step in getattr(trajectory, "steps", []):
        kind = getattr(getattr(step, "kind", None), "value", getattr(step, "kind", None))
        payload = getattr(step, "payload", {}) or {}
        if kind == "tool_call":
            pending = payload.get("name")
        elif kind == "tool_result" and pending == "summarize":
            result = payload.get("result", {})
            if isinstance(result, dict) and "answer" in result:
                answer = result["answer"]
            pending = None
    return answer


# ---------------------------------------------------------------------------
# Policies and agent
# ---------------------------------------------------------------------------

# The user's request: deliberately vague ("last quarter", "brought in"), but it
# carries today's date the way a real query would. The four implicit rules are
# what a good prompt must pin down.
TASK_REQUEST = (
    "Today's date is 2026-04-15. Who were our top 3 reps last quarter and how much "
    "did they bring in? Give me the numbers."
)

POLICY_GENESIS = (
    "You are a data analyst. Answer the user's question using the available tools, "
    "then report the result."
)

# The oracle pins all four implicit rules; the search must discover this.
POLICY_ORACLE = (
    "You are a sales analyst answering questions from a company database using the "
    "query tools. For a 'top reps last quarter' question, follow these rules exactly: "
    "(1) revenue is NET per line item: qty * unit_price * (1 - discount) - refund. "
    "Subtract the refund column; do not ignore it. (2) 'last quarter' means the "
    "single quarter immediately before today's quarter. Today is 2026-04-15, which "
    "is in 2026-Q2, so last quarter is 2026-Q1 (January through March 2026). It does "
    "NOT mean the fourth quarter of the year, and NOT the most recent quarter that "
    "happens to have data. Filter to quarter == '2026-Q1'. (3) EXCLUDE orders whose "
    "status is 'cancelled' OR 'returned'; only count status 'ok'. (4) EXCLUDE orders "
    "whose channel is 'internal' (inter-company transfers, not real revenue); keep "
    "only channel 'direct'. (5) EXCLUDE orders whose payment is 'pending'; only count "
    "payment 'paid'. (6) Group the totals by sales rep (rep_name), not by customer or "
    "product. So: join orders to line_items for revenue and to reps for the rep name, "
    "add a quarter column, keep only quarter '2026-Q1', status 'ok', channel 'direct', "
    "and payment 'paid', compute net line revenue (with the refund subtracted), sum it "
    "by rep_name, take the top 3, and report with summarize(handle, 'rep_name', "
    "'line_revenue')."
)


def schema_description() -> str:
    """The database schema, for whoever is improving the agent's prompt. A real
    prompt engineer would know this; from it plus the scorer's feedback, the search
    can infer which rules the prompt is missing without being handed the answer."""
    return (
        "Database schema:\n"
        "- orders(order_id, customer_id, rep_id, order_date, status); status is one of "
        "'ok', 'cancelled', 'returned'.\n"
        "- line_items(order_id, product_id, qty, unit_price, discount, refund); discount is a "
        "fraction 0..1, refund is a dollar amount on that line.\n"
        "- reps(rep_id, rep_name); customers(customer_id, customer_name); products(product_id, product_name).\n"
        "Tools: load_table, filter_rows, join_tables, compute_column, date_bucket, aggregate, "
        "sort_top_n, summarize."
    )


def build_revenue_policy(content: str) -> Artifact:
    return genesis(id=SYSTEM_PROMPT_ID, kind=Subtype.PROMPT, content=content, created_by="human")


def build_revenue_agent(system_prompt: str | Artifact, *, model: str = DEFAULT_MODEL,
                        temperature: float | None = None) -> Agent:
    sp = system_prompt if isinstance(system_prompt, Artifact) else build_revenue_policy(system_prompt)
    # The chain runs ~12 tool calls; give it headroom so a complete answer fits.
    return Agent(system_prompt=sp, tools=make_revenue_tools(), model=model,
                 temperature=temperature, max_iterations=16, max_tool_calls=32)
