"""D6's desk for decision programs: a billing world, its read tools, the trusted procedure slot, and issue_credit."""

from __future__ import annotations

from helix.metacognition.decide import Desk

from examples.billing.world import BillingWorld


def desk() -> Desk:
    """In a ProgramTier the loop calls the procedure the agent has learned, so none is given here."""
    return Desk(world=BillingWorld, reads=lambda world: {"read_account": world.read_account, "read_policy": world.read_policy},
                procedures=lambda world: {},
                actions={"issue_credit": lambda world, account_id, amount_usd: world.issue_credit(account_id, amount_usd)},
                verdict=lambda world, reply: world.check())
