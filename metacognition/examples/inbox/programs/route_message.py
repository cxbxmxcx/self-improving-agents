"""Files one ticket with no language model. Jev names the topic in two steps; the routing function says the rest."""

SURE = 0.5
FAMILIES = {
    "cards": ["card_arrival", "card_linking", "card_delivery_estimate", "card_not_working", "lost_or_stolen_card",
              "pin_blocked", "contactless_not_working", "getting_virtual_card", "get_physical_card",
              "getting_spare_card", "order_physical_card", "virtual_card_not_working", "get_disposable_virtual_card",
              "activate_my_card", "card_about_to_expire", "card_swallowed", "compromised_card", "change_pin",
              "apple_pay_or_google_pay", "lost_or_stolen_phone", "supported_cards_and_currencies",
              "visa_or_mastercard", "disposable_card_limits"],
    "card payments": ["card_payment_wrong_exchange_rate", "extra_charge_on_statement", "card_payment_fee_charged",
                      "card_payment_not_recognised", "pending_card_payment", "declined_card_payment",
                      "reverted_card_payment?", "transaction_charged_twice", "request_refund",
                      "Refund_not_showing_up", "direct_debit_payment_not_recognised", "card_acceptance"],
    "cash and ATMs": ["pending_cash_withdrawal", "wrong_amount_of_cash_received", "declined_cash_withdrawal",
                      "cash_withdrawal_not_recognised", "wrong_exchange_rate_for_cash_withdrawal",
                      "cash_withdrawal_charge", "atm_support"],
    "transfers": ["cancel_transfer", "transfer_not_received_by_recipient", "declined_transfer", "pending_transfer",
                  "transfer_timing", "beneficiary_not_allowed", "transfer_fee_charged", "receiving_money",
                  "failed_transfer", "transfer_into_account", "balance_not_updated_after_bank_transfer"],
    "top-ups": ["automatic_top_up", "top_up_by_bank_transfer_charge", "pending_top_up", "top_up_reverted",
                "topping_up_by_card", "top_up_by_cash_or_cheque", "verify_top_up", "top_up_failed",
                "top_up_by_card_charge", "balance_not_updated_after_cheque_or_cash_deposit", "top_up_limits"],
    "currency": ["exchange_via_app", "exchange_charge", "exchange_rate", "fiat_currency_support"],
    "account and identity": ["edit_personal_details", "unable_to_verify_identity", "verify_my_identity",
                             "verify_source_of_funds", "terminate_account", "passcode_forgotten", "country_support",
                             "age_limit", "why_verify_identity"],
}
FAMILY = {"type": "choice", "instructions": "Which part of the bank's services is `request` about?",
          "criteria": {"cards": "ordering, receiving, activating, replacing or using a card, its PIN, a lost or "
                                "stolen card or phone, virtual and disposable cards, Apple Pay or Google Pay",
                       "card payments": "a payment made with a card or a direct debit: declined, pending, reverted, "
                                        "charged twice, not recognised, a fee, a refund",
                       "cash and ATMs": "cash withdrawn from an ATM: declined, pending, the wrong amount, a charge",
                       "transfers": "sending or receiving money by bank transfer",
                       "top-ups": "adding money to the account by card, bank transfer, cash or cheque",
                       "currency": "exchanging currencies, exchange rates and which currencies are held",
                       "account and identity": "personal details, identity checks, passcodes, closing the account, "
                                               "who may open one and where"}}
MEMORY = ("Read the routing policy, pick the topic in its table that the message is about, call route_ticket with "
          "that topic and the customer id, and file one ticket with what it says, or none when it says no ticket.")


def topic_question(family):
    criteria = {}
    for topic in FAMILIES[family]:
        criteria[topic] = topic.replace("_", " ").replace("?", "")
    return {"type": "choice", "instructions": "Which of these topics is `request` about?", "criteria": criteria}


async def decide(request, kit):
    ids = kit.find("C[0-9]{5}", request)
    if len(ids) != 1:
        return {"handoff": "no customer id in the message", "route": "none", "memory": MEMORY}
    judged = await kit.ask({"request": request}, {"family": FAMILY})
    family = judged["family"]["choice"]
    if family not in FAMILIES or judged["family"]["confidence"] < SURE:
        return {"handoff": "not sure which part of the bank this is about", "route": "none", "memory": MEMORY}
    judged = await kit.ask({"request": request}, {"topic": topic_question(family)})
    topic = judged["topic"]["choice"]
    if topic not in FAMILIES[family] or judged["topic"]["confidence"] < SURE:
        return {"handoff": "not sure which topic this is", "route": "none", "memory": MEMORY}
    routed = await kit.procedure("route_ticket", topic=topic, customer_id=ids[0])
    if not isinstance(routed, dict) or "ticket" not in routed:
        return {"handoff": "the routing function gave no answer", "route": "none", "memory": MEMORY}
    if not routed["ticket"]:
        return {"handoff": "the help pages answer this, so there is no ticket", "route": "none", "read_only": True,
                "memory": "Answer with a short pointer to the help pages for " + topic + ", and file nothing."}
    return {"action": "file_ticket",
            "arguments": {"topic": topic, "team": routed["team"], "urgent": bool(routed["urgent"])},
            "reply": "Filed for " + str(routed["team"]) + (", urgent." if routed["urgent"] else ".")}
