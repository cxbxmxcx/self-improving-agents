"""The accounts-payable desk, written on metacog: tools, the two procedures worth learning, and how runs are judged.

Each procedure answers in its action tool's own terms (D2's lesson): `parse_invoice` returns record_invoice's
arguments and `amount_payable` returns schedule_payment's, so the agent only copies them across.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from helix.memory.code_skills import skill_tool
from helix.metacognition import procedure

from examples.invoices.world import InvoiceWorld, task_by_id
from metacog import Agent, RunContextWrapper, function_tool

MODEL = "gpt-5.4-mini"
PROMPT = """You work accounts payable for several client companies. Each request is one of three kinds of work:
- a vendor invoice document to enter: read the document and record the invoice with record_invoice;
- an entered invoice to pay: read the invoice, its purchase order and the client's terms, work out the payment,
  and schedule it with schedule_payment;
- a question about a vendor: look the vendor up and answer. Don't record or schedule anything for a question.
Act once, then reply with what you did."""
MEMORIES = Path(__file__).parent / "memories"
# The hand-written procedures as they stand: the parser reads Kestrel's European layout too.
PARSER = (MEMORIES / "parse_invoice.py").read_text(encoding="utf-8")
PAYABLE = (MEMORIES / "amount_payable.py").read_text(encoding="utf-8")
EUROPEAN = 'european = lines[0].endswith("GMBH")'


def before_layout_change(source: str) -> str:
    """The parser as written before Kestrel's European layout existed: the same code without that one test, so
    it reads the new layout's dates and amounts the old way, wrongly and without failing."""
    if EUROPEAN not in source:
        raise ValueError("the parser doesn't test for Kestrel's European layout")
    return source.replace(EUROPEAN, "european = False").replace(
        "every vendor's layout, Kestrel's European one included",
        "every vendor's layout as it was before Kestrel's European one")


# Where D3's online loop starts: the parser written before the layout change.
PARSER_BEFORE = before_layout_change(PARSER)


@function_tool(read_only=True)
def read_document(ctx: RunContextWrapper[InvoiceWorld], doc_id: str) -> dict:
    """Read a scanned vendor document: an invoice's text, in the vendor's own layout."""
    return ctx.context.read_document(doc_id)


@function_tool(read_only=True)
def read_invoice(ctx: RunContextWrapper[InvoiceWorld], invoice_id: str) -> dict:
    """Read an invoice already entered in the ledger: vendor, client, purchase order, date, lines (SKU,
    category, quantity, unit price), freight, and the vendor's payment terms."""
    return ctx.context.read_invoice(invoice_id)


@function_tool(read_only=True)
def read_purchase_order(ctx: RunContextWrapper[InvoiceWorld], po_number: str) -> dict:
    """Read a purchase order: its client, and for each SKU the quantity ordered and the quantity delivered."""
    return ctx.context.read_purchase_order(po_number)


@function_tool(read_only=True)
def read_client_terms(ctx: RunContextWrapper[InvoiceWorld], client_id: str) -> dict:
    """Read a client's accounts-payable terms: its sales tax rate, taxed categories, and payment rules."""
    return ctx.context.read_client_terms(client_id)


@function_tool(read_only=True)
def read_vendor(ctx: RunContextWrapper[InvoiceWorld], vendor: str) -> dict:
    """Look up a vendor by name or id: its contacts, including where remittance advice goes."""
    return ctx.context.read_vendor(vendor)


@function_tool
def record_invoice(ctx: RunContextWrapper[InvoiceWorld], doc_id: str, invoice_number: str, po_number: str,
                   invoice_date: str, subtotal_usd: float, freight_usd: float, discount_percent: float,
                   discount_days: int, net_days: int) -> dict:
    """Enter a vendor invoice in the ledger from its document. The date is YYYY-MM-DD; the subtotal is the goods
    before freight; discount_percent and discount_days are 0 when the vendor offers no early-payment discount."""
    return ctx.context.record_invoice(doc_id=doc_id, invoice_number=invoice_number, po_number=po_number,
                                      invoice_date=invoice_date, subtotal_usd=subtotal_usd, freight_usd=freight_usd,
                                      discount_percent=discount_percent, discount_days=discount_days,
                                      net_days=net_days)


@function_tool
def schedule_payment(ctx: RunContextWrapper[InvoiceWorld], invoice_id: str, amount_usd: float, pay_on: str) -> dict:
    """Schedule the payment of an entered invoice: the amount in US dollars and the date to pay, YYYY-MM-DD."""
    return ctx.context.schedule_payment(invoice_id, amount_usd, pay_on)


READS = [read_document, read_invoice, read_purchase_order, read_client_terms, read_vendor]


@procedure(api=[read_document])
def parse_invoice(doc_id: str) -> dict:
    """Read a vendor invoice document, in whatever layout its vendor uses, and extract what the ledger records.

    Returns: record_invoice's arguments: {"doc_id": str, "invoice_number": str, "po_number": str, "invoice_date":
    "YYYY-MM-DD", "subtotal_usd": float, "freight_usd": float, "discount_percent": float, "discount_days": int,
    "net_days": int}; the subtotal is the goods before freight, rounded to the cent.
    """
    ...


@procedure(api=[read_invoice, read_purchase_order, read_client_terms])
def amount_payable(invoice_id: str) -> dict:
    """Work out what to pay on an entered invoice, and when, under its client's accounts-payable terms.

    Returns: schedule_payment's arguments: {"invoice_id": str, "amount_usd": float, "pay_on": "YYYY-MM-DD"}.
    """
    ...


@procedure(api=[read_invoice, read_purchase_order, read_client_terms], name="amount_payable")
def amount_payable_to_act_on(invoice_id: str) -> dict:
    """Work out what to pay on an entered invoice, and when, under its client's accounts-payable terms. It reads
    the invoice, its purchase order and the client's terms itself, including conventions the client never wrote
    down: call it first and pass its answer to schedule_payment, without working the amount out again.

    Returns: schedule_payment's arguments: {"invoice_id": str, "amount_usd": float, "pay_on": "YYYY-MM-DD"}.
    """
    ...


PROCEDURES = [parse_invoice, amount_payable]
# D4: the payable's contract in the tools' terms, and the same contract telling the agent to act on its answer
# (D2's fix for an agent that overrules a right procedure). Same name, signature and return.
CONTRACTS = {"tool": PROCEDURES, "act": [parse_invoice, amount_payable_to_act_on]}
KIND_OF = {"parse_invoice": "intake", "amount_payable": "payment"}  # the work each procedure does


def ap_desk(model: str = MODEL, **kwargs: Any) -> Agent:
    return Agent(name="Accounts payable desk", instructions=PROMPT, model=model,
                 tools=[*READS, record_invoice, schedule_payment], **kwargs)


def check(ctx: RunContextWrapper[InvoiceWorld], result: Any) -> dict:
    """The audit of what was done with the request."""
    return ctx.context.check(None if result is None else result.final_output)


class WorldAPI:
    """What a procedure reads in function-level practice: the case's own world, read-only."""

    def __init__(self, world: InvoiceWorld) -> None:
        self.read_document = world.read_document
        self.read_invoice = world.read_invoice
        self.read_purchase_order = world.read_purchase_order
        self.read_client_terms = world.read_client_terms
        self.read_vendor = world.read_vendor


def procedure_of(source: str) -> str | None:
    """Which procedure a source defines."""
    return next((name for name in KIND_OF if f"def {name}(" in source), None)


async def run_on_case(source: str, task: dict[str, Any]) -> Any:
    """A procedure's answer on one case, under the code-skill guard."""
    name = procedure_of(source) or "parse_invoice"
    argument = ({"doc_id": task.get("doc_id", "")} if name == "parse_invoice" else
                {"invoice_id": task["invoice"]["invoice_id"]})
    try:
        tool = skill_tool(name, "", source, bind={"api": WorldAPI(InvoiceWorld(task))})
        return await tool.call(**argument)
    except Exception as exc:  # noqa: BLE001 - refused by the guard or not callable with the contract's arguments
        return {"error": f"{name} could not be called: {type(exc).__name__}: {exc}"}


RECORD_FIELDS = ("doc_id", "invoice_number", "po_number", "invoice_date", "subtotal_usd", "freight_usd",
                 "discount_percent", "discount_days", "net_days")


def apply(world: InvoiceWorld, name: str, answer: dict[str, Any]) -> None:
    """Carry out a procedure's answer the way the agent would: pass it to its action tool."""
    if name == "parse_invoice":
        world.record_invoice(**{key: answer.get(key) for key in RECORD_FIELDS})
    else:
        world.schedule_payment(answer.get("invoice_id"), answer.get("amount_usd"), answer.get("pay_on"))


def _not_mine(name: str | None, task: dict[str, Any]) -> str | None:
    if name is None:
        return "the source defines neither procedure"
    if KIND_OF[name] != task["kind"]:
        return f"{task['id']} is {task['kind']} work, not {name}'s: nothing to try"
    return None


async def practice(source: str, case_id: str) -> tuple[float, str, str]:
    """Chapter 6's act for this world: run the function on the case, carry out its answer, ask the audit.

    One Gate and one learner serve both procedures, so a case of the other procedure's kind has nothing to try
    and passes; with the gate's bar at 1.0 a version must then pass every case of its own kind."""
    task = task_by_id(case_id)
    name = procedure_of(source)
    if (skip := _not_mine(name, task)) is not None:
        return (0.0 if name is None else 1.0), skip, ""
    answer = await run_on_case(source, task)
    observed = f"returned {json.dumps(answer, default=str)[:600]}"
    if task["kind"] == "intake":  # what the function read, so a lesson can describe the layout it met
        observed = f"read the document:\n{task['document']}\n{observed}"
    if not isinstance(answer, dict) or "error" in answer:
        return 0.0, f"the function failed: {answer.get('error') if isinstance(answer, dict) else answer!r}", observed
    world = InvoiceWorld(task)
    apply(world, name, answer)
    verdict = world.check()
    return (1.0 if verdict["success"] else 0.0), verdict["reason"], observed


def _agrees(got: Any, want: Any) -> bool:
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        try:
            return abs(float(got) - float(want)) <= 0.01 + 1e-9
        except (TypeError, ValueError):
            return False
    return str(got).strip().upper() == str(want).strip().upper()


async def labeled_practice(source: str, case_id: str) -> tuple[float, str, str]:
    """The gate's practice: a case file's labeled case, where the label is the action's right arguments. The
    function's answer is compared with the label, field by field; no audit is asked."""
    task = task_by_id(case_id)
    name = procedure_of(source)
    if (skip := _not_mine(name, task)) is not None:
        return (0.0 if name is None else 1.0), skip, ""
    answer = await run_on_case(source, task)
    observed = f"returned {json.dumps(answer, default=str)[:600]}"
    if not isinstance(answer, dict) or "error" in answer:
        return 0.0, "the function failed", observed
    wrong = [key for key, want in task["expected"].items() if not _agrees(answer.get(key), want)]
    if wrong:
        return 0.0, f"differs from the label in {', '.join(wrong)}", observed
    return 1.0, "matches the label", observed
