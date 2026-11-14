"""Serves payment and intake requests with no language model. Jev says what the request asks; questions about
vendors, which need a written answer, go to the agent."""

INVOICE = r"\bINV-\d+\b"
DOCUMENT = r"\bDOC-\d+\b"
WORK = {
    "type": "choice",
    "instructions": "What does `request` ask the accounts payable desk to do?",
    "criteria": {
        "payment": "Pay, or schedule the payment of, an invoice already in the ledger",
        "intake": "Enter a vendor invoice document into the ledger",
        "question": "Answer a question, such as one about a vendor",
    },
}
FITS = {"type": "noul", "instructions": "Does `request` ask the accounts payable desk to pay an invoice, enter an "
                                        "invoice document, or answer a question about a vendor?"}
SURE = 0.7


async def decide(request, kit):
    answers = await kit.ask({"request": request}, {"work": WORK, "fits": FITS})
    work = answers["work"]
    if answers["fits"]["noul"] < SURE or work["probabilities"][work["choice"]] < SURE:
        return {"handoff": "Jev isn't sure what the request asks"}
    if work["choice"] == "payment":
        invoices = kit.find(INVOICE, request)
        if len(invoices) != 1:
            return {"handoff": "the request names " + str(len(invoices)) + " invoices, not one"}
        answer = await kit.procedure("amount_payable", invoice_id=invoices[0])
        if not isinstance(answer, dict) or "amount_usd" not in answer:
            return {"handoff": "the payable procedure gave no amount"}
        return {"action": "schedule_payment",
                "arguments": {"invoice_id": answer["invoice_id"], "amount_usd": answer["amount_usd"],
                              "pay_on": answer["pay_on"]},
                "reply": "I scheduled " + invoices[0] + " for payment on " + answer["pay_on"] + "."}
    if work["choice"] == "intake":
        documents = kit.find(DOCUMENT, request)
        if len(documents) != 1:
            return {"handoff": "the request names " + str(len(documents)) + " documents, not one"}
        record = await kit.procedure("parse_invoice", doc_id=documents[0])
        if not isinstance(record, dict) or "invoice_number" not in record:
            return {"handoff": "the parser gave no record"}
        return {"action": "record_invoice", "arguments": record,
                "reply": "I entered " + documents[0] + " in the ledger."}
    return {"handoff": "a question needs a written answer, so the agent takes it"}
