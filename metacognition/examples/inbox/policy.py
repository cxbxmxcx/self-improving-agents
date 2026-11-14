"""The bank's routing policy: which team a topic goes to, what is urgent, and who gets premium care.

The topics are Banking77's 77 labels, as the dataset's own annotators assigned them. The teams and rules are ours, and
they are what a support desk's ticket must get right: the team that handles it and whether it is urgent. Some topics
are answered from the help pages and get no ticket at all.

Three versions of the policy are in force at different times in the stream. A function that learned one version by
heart files some tickets wrongly under the next, as a function that read Dana's note in Chapter 6 did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

HELP = "Help pages"  # not a team: these topics are answered with a link to the help pages and no ticket
PREMIUM = "Premium care"
DISPUTES = "Disputes"

TEAMS: dict[str, tuple[str, ...]] = {
    "Cards": (
        "card_arrival", "card_linking", "card_delivery_estimate", "card_not_working", "lost_or_stolen_card",
        "pin_blocked", "contactless_not_working", "getting_virtual_card", "get_physical_card", "getting_spare_card",
        "order_physical_card", "virtual_card_not_working", "get_disposable_virtual_card", "activate_my_card",
        "card_about_to_expire", "card_swallowed", "compromised_card", "change_pin", "apple_pay_or_google_pay",
        "lost_or_stolen_phone"),
    "Card payments": (
        "card_payment_wrong_exchange_rate", "extra_charge_on_statement", "card_payment_fee_charged",
        "card_payment_not_recognised", "pending_card_payment", "declined_card_payment", "reverted_card_payment?",
        "transaction_charged_twice", "request_refund", "Refund_not_showing_up", "direct_debit_payment_not_recognised"),
    "Cash and ATMs": (
        "pending_cash_withdrawal", "wrong_amount_of_cash_received", "declined_cash_withdrawal",
        "cash_withdrawal_not_recognised", "wrong_exchange_rate_for_cash_withdrawal", "cash_withdrawal_charge"),
    "Transfers": (
        "cancel_transfer", "transfer_not_received_by_recipient", "declined_transfer", "pending_transfer",
        "transfer_timing", "beneficiary_not_allowed", "transfer_fee_charged", "receiving_money", "failed_transfer",
        "transfer_into_account", "balance_not_updated_after_bank_transfer"),
    "Top-ups": (
        "automatic_top_up", "top_up_by_bank_transfer_charge", "pending_top_up", "top_up_reverted",
        "topping_up_by_card", "top_up_by_cash_or_cheque", "verify_top_up", "top_up_failed", "top_up_by_card_charge",
        "balance_not_updated_after_cheque_or_cash_deposit"),
    "Currency": ("exchange_via_app", "exchange_charge"),
    "Account and identity": (
        "edit_personal_details", "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
        "terminate_account", "passcode_forgotten"),
    HELP: (
        "exchange_rate", "supported_cards_and_currencies", "fiat_currency_support", "country_support", "age_limit",
        "visa_or_mastercard", "card_acceptance", "atm_support", "top_up_limits", "disposable_card_limits",
        "why_verify_identity"),
}
URGENT = ("lost_or_stolen_card", "compromised_card", "card_swallowed", "lost_or_stolen_phone",
          "card_payment_not_recognised", "cash_withdrawal_not_recognised", "direct_debit_payment_not_recognised")
# The topics the Disputes team takes over when the second version of the policy comes in.
DISPUTED = ("transaction_charged_twice", "card_payment_not_recognised", "cash_withdrawal_not_recognised",
            "direct_debit_payment_not_recognised", "request_refund", "Refund_not_showing_up",
            "wrong_amount_of_cash_received", "extra_charge_on_statement")
TOPICS = tuple(topic for topics in TEAMS.values() for topic in topics)
TOP_UPS = TEAMS["Top-ups"] + ("top_up_limits",)  # the work that arrives later in the stream


@dataclass(frozen=True)
class Policy:
    """One version of the routing policy."""

    key: str
    teams: dict[str, str]  # topic -> team, or HELP
    urgent: tuple[str, ...]
    premium_plans: tuple[str, ...]  # whose tickets go to Premium care, unless urgent or kept by `kept_teams`
    kept_teams: tuple[str, ...] = ()  # teams whose tickets stay with them for premium customers too
    text: str = field(default="", compare=False)

    def route(self, topic: str, plan: str) -> dict[str, Any]:
        """What a ticket for `topic` from a customer on `plan` must say: its team and urgency, or no ticket."""
        team = self.teams[topic]
        if team == HELP:
            return {"ticket": False, "team": None, "urgent": False}
        urgent = topic in self.urgent
        if plan in self.premium_plans and not urgent and team not in self.kept_teams:
            team = PREMIUM
        return {"ticket": True, "team": team, "urgent": urgent}


def _teams(moved: dict[str, str] | None = None) -> dict[str, str]:
    out = {topic: team for team, topics in TEAMS.items() for topic in topics}
    out.update(moved or {})
    return out


def _text(policy_key: str, teams: dict[str, str], urgent: tuple[str, ...], premium: tuple[str, ...],
          kept: tuple[str, ...]) -> str:
    """The policy as the desk's staff read it: the rules, then the table."""
    who = " and ".join(f"{plan}" for plan in premium)
    kept_line = (f" Tickets for {', '.join(kept)} stay with that team for every customer." if kept else "")
    rules = [
        f"Routing policy {policy_key}.",
        "Every message about one of the topics below gets one ticket for the team the table names, with no other "
        "action. A topic the table sends to Help pages is answered with a link to the help pages and gets no ticket.",
        f"These topics are urgent, and their tickets are filed urgent: {', '.join(urgent)}. Every other ticket is not "
        "urgent.",
        f"Tickets from customers on the {who} plan go to Premium care instead of the team in the table, except urgent "
        f"tickets, which stay with their team.{kept_line}",
        "A message that is not about the bank's own services gets a short reply and no ticket.",
    ]
    table = "\n".join(f"{topic}: {team}" for topic, team in sorted(teams.items()))
    return "\n".join(rules) + "\n\nTopic: team\n" + table


def _policy(key: str, moved: dict[str, str] | None, premium: tuple[str, ...], kept: tuple[str, ...]) -> Policy:
    teams = _teams(moved)
    return Policy(key, teams, URGENT, premium, kept, _text(key, teams, URGENT, premium, kept))


# The first version; the second moves disputed payments to a new team that keeps them for premium customers too; the
# third keeps that and narrows premium care to the metal plan.
FIRST = _policy("2026-1", None, ("premium", "metal"), ())
SECOND = _policy("2026-2", {topic: DISPUTES for topic in DISPUTED}, ("premium", "metal"), (DISPUTES,))
THIRD = _policy("2026-3", {topic: DISPUTES for topic in DISPUTED}, ("metal",), (DISPUTES,))
POLICIES = {policy.key: policy for policy in (FIRST, SECOND, THIRD)}
