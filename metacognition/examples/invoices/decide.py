"""D3's desk for decision programs: an invoice world, its read tools, the trusted procedures, and the actions."""

from __future__ import annotations

from typing import Any

from helix.memory.code_skills import skill_tool
from helix.metacognition.decide import Desk

from examples.invoices.agent import PARSER, PAYABLE, RECORD_FIELDS, WorldAPI
from examples.invoices.world import InvoiceWorld


def desk(parser: str = PARSER, payable: str = PAYABLE) -> Desk:
    """`parser` and `payable` are the trusted procedure versions a program may call (hand-written by default)."""

    def procedures(world: InvoiceWorld) -> dict[str, Any]:
        def bound(name: str, source: str) -> Any:
            async def call(**arguments: Any) -> Any:
                tool = skill_tool(name, "", source, bind={"api": WorldAPI(world)})
                return await tool.call(**arguments)

            return call

        return {"parse_invoice": bound("parse_invoice", parser), "amount_payable": bound("amount_payable", payable)}

    def reads(world: InvoiceWorld) -> dict[str, Any]:
        return {name: getattr(world, name) for name in ("read_document", "read_invoice", "read_purchase_order",
                                                        "read_client_terms", "read_vendor")}

    return Desk(world=InvoiceWorld, reads=reads, procedures=procedures,
                actions={"record_invoice": lambda world, **record: world.record_invoice(
                             **{key: record.get(key) for key in RECORD_FIELDS}),
                         "schedule_payment": lambda world, invoice_id, amount_usd, pay_on: world.schedule_payment(
                             invoice_id, amount_usd, pay_on)},
                verdict=lambda world, reply: world.check(reply))
