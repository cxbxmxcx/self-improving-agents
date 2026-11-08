PAYMENTPHRASES = ["set up", "schedule", "pay", "payment", "owe", "when"]
RECORDPHRASES = ["enter", "record", "ledger", "post"]
DISALLOWPHRASES = ["cancel", "dispute", "status", "delete"]


def hasAny(text, phrases):
    for phrase in phrases:
        if phrase in text:
            return True
    return False


def digitsOnly(text):
    out = ""
    for ch in str(text):
        if ch >= "0" and ch <= "9":
            out += ch
    return out


def uniqueList(items):
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def normalizeId(raw, kind):
    text = str(raw).lower().strip()
    if kind == "invoice":
        prefix = "inv"
        word = "invoice"
    else:
        prefix = "doc"
        word = "document"

    if text.startswith(word):
        text = text[len(word):].strip()
    if text.startswith(prefix):
        text = text[len(prefix):].strip()

    text = text.replace("-", " ")
    text = text.replace("_", " ")
    digits = digitsOnly(text)
    if digits == "":
        return ""
    return prefix.upper() + "-" + digits


def collectIds(kit, text, kind):
    if kind == "invoice":
        patterns = [
            "invoice[ ]*inv[- ]*[0-9]+",
            "invoice[ ]*[0-9]+",
            "inv[- ]*[0-9]+",
            "inv[ ]*[0-9]+"
        ]
    else:
        patterns = [
            "document[ ]*doc[- ]*[0-9]+",
            "document[ ]*[0-9]+",
            "doc[- ]*[0-9]+",
            "doc[ ]*[0-9]+"
        ]

    ids = []
    for pattern in patterns:
        found = kit.find(pattern, text)
        for item in found:
            clean = normalizeId(item, kind)
            if clean != "" and clean not in ids:
                ids.append(clean)
    return ids


async def pickId(kit, request, ids, kind):
    ids = uniqueList(ids)
    if len(ids) == 0:
        return None
    if len(ids) == 1:
        return ids[0]

    criteria = {}
    for item in ids:
        criteria[item] = "candidate " + kind + " id"
    criteria["none"] = "none of these"

    state = {"request": request, "kind": kind, "candidates": ids}
    questions = {
        "pick": {
            "type": "choice",
            "instructions": "Select the most likely " + kind + " id the user wants acted on. Choose none if none of these is the target.",
            "criteria": criteria
        }
    }

    answer = await kit.ask(state, questions)
    pick = answer["pick"]
    if pick["choice"] == "none":
        return None
    if pick["choice"] not in ids:
        return None
    if pick.get("confidence", 0.0) < 0.7:
        return None
    return pick["choice"]


async def resolvePayable(kit, invoiceId):
    variants = [invoiceId]
    digits = digitsOnly(invoiceId)
    if digits != "" and digits != invoiceId:
        variants.append(digits)

    for item in variants:
        try:
            data = await kit.procedure("amount_payable", invoice_id=item)
        except Exception:
            data = None
        if isinstance(data, dict) and "error" not in data and "amount_usd" in data and "pay_on" in data:
            outId = item
            if "invoice_id" in data and isinstance(data["invoice_id"], str) and data["invoice_id"] != "":
                outId = data["invoice_id"]
            return {
                "invoice_id": outId,
                "amount_usd": float(data["amount_usd"]),
                "pay_on": str(data["pay_on"])
            }
    return None


async def resolveRecord(kit, docId):
    variants = [docId]
    digits = digitsOnly(docId)
    if digits != "" and digits != docId:
        variants.append(digits)

    for item in variants:
        try:
            record = await kit.procedure("parse_invoice", doc_id=item)
        except Exception:
            record = None
        if isinstance(record, dict) and "error" not in record:
            return record
    return None


async def decide(request, kit):
    text = str(request)
    low = text.lower()

    for phrase in DISALLOWPHRASES:
        if phrase in low:
            return {"handoff": "This request is about a status, cancellation, dispute, or delete action, so it should be handed off."}

    invoiceIds = collectIds(kit, low, "invoice")
    docIds = collectIds(kit, low, "document")

    paymentIntent = len(invoiceIds) > 0 and hasAny(low, PAYMENTPHRASES)
    recordIntent = len(docIds) > 0 and hasAny(low, RECORDPHRASES)

    if paymentIntent and recordIntent:
        return {"handoff": "This request appears to mix payment scheduling and ledger entry, so it should be handed off."}

    if paymentIntent:
        invoiceId = await pickId(kit, text, invoiceIds, "invoice")
        if invoiceId is None:
            return {"handoff": "I could not identify which invoice should be scheduled."}

        payable = await resolvePayable(kit, invoiceId)
        if payable is None:
            return {"handoff": "I could not get the payable amount for " + invoiceId + "."}

        return {
            "action": "schedule_payment",
            "arguments": {
                "invoice_id": payable["invoice_id"],
                "amount_usd": payable["amount_usd"],
                "pay_on": payable["pay_on"]
            },
            "reply": "Scheduling payment for " + payable["invoice_id"] + "."
        }

    if recordIntent:
        docId = await pickId(kit, text, docIds, "document")
        if docId is None:
            return {"handoff": "I could not identify which document should be entered."}

        record = await resolveRecord(kit, docId)
        if record is None:
            return {"handoff": "I could not parse " + docId + " for ledger entry."}

        return {
            "action": "record_invoice",
            "arguments": record,
            "reply": "Recording " + docId + " into the ledger."
        }

    if "vendor" in low:
        return {"handoff": "This is about a vendor rather than a payment scheduling or invoice-entry task."}

    return {"handoff": "This request is not a payment scheduling or invoice-entry task."}
