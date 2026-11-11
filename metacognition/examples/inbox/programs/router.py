"""Picks the memory that serves a message. Jev judges whether it is about the bank's own services at all."""

SURE = 0.7
ABOUT_THE_BANK = {"type": "noul",
                  "instructions": "Is `request` about the bank's own cards, card payments, cash withdrawals, "
                                  "transfers, top-ups, currency exchange, or the customer's account?"}


async def decide(request, kit):
    judged = await kit.ask({"request": request}, {"bank": ABOUT_THE_BANK})
    if judged["bank"]["noul"] < SURE:
        return {"serve": "not_for_the_bank"}
    return {"serve": "route_message"}
