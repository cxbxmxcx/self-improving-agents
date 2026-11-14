JOB_SCHEDULE = "schedule_payment"
JOB_RECORD = "record_invoice"
JOB_NONE = "none"

TARGET_NONE = "none"

INVOICE_PATTERNS = ["INV-[0-9]+", "[Ii]nvoice[^0-9]{0,20}[0-9]+"]
DOC_PATTERNS = ["DOC-[0-9]+", "[Dd]ocument[^0-9]{0,20}[0-9]+"]


def clean_value(value):
    if isinstance(value, float):
        whole = int(value)
        if whole == value:
            return whole
        return round(value, 2)
    return value


def clean_record(record):
    result = dict(record)
    keys = ["freight_usd", "discount_percent", "discount_days", "net_days", "subtotal_usd"]
    for key in keys:
        if key in result:
            result[key] = clean_value(result[key])
    return result


def unique_matches(patterns, text, kit):
    items = []
    seen = {}
    for pattern in patterns:
        matches = kit.find(pattern, text)
        for match in matches:
            if match not in seen:
                seen[match] = True
                items.append(match)
    return items


def add_unique(items, value):
    if value not in items:
        items.append(value)
    return items


def normalize_options(target, kind, kit):
    options = []
    if kind == "invoice":
        prefix = "INV-"
        full_matches = kit.find("[Ii][Nn][Vv]-[0-9]+", target)
    else:
        prefix = "DOC-"
        full_matches = kit.find("[Dd][Oo][Cc]-[0-9]+", target)

    if len(full_matches) > 0:
        full = full_matches[0].upper()
        options = add_unique(options, full)
        digits = kit.find("[0-9]+", full)
        if len(digits) > 0:
            num = digits[0]
            if len(num) < 3:
                options = add_unique(options, prefix + num.zfill(3))
            options = add_unique(options, prefix + num)
            options = add_unique(options, num)
        return options

    digits = kit.find("[0-9]+", target)
    if len(digits) > 0:
        num = digits[0]
        if len(num) < 3:
            options = add_unique(options, prefix + num.zfill(3))
        options = add_unique(options, prefix + num)
        options = add_unique(options, num)
    return options


def valid_answer(result, keys):
    if not isinstance(result, dict):
        return False
    if "error" in result:
        return False
    for key in keys:
        if key not in result:
            return False
    return True


async def try_amount_payable(invoice_options, kit):
    for invoice_id in invoice_options:
        try:
            result = await kit.procedure("amount_payable", invoice_id=invoice_id)
        except Exception:
            result = None
        if valid_answer(result, ["invoice_id", "amount_usd", "pay_on"]):
            return result, invoice_id
    return None, ""


async def try_parse_invoice(doc_options, kit):
    for doc_id in doc_options:
        try:
            result = await kit.procedure("parse_invoice", doc_id=doc_id)
        except Exception:
            result = None
        if valid_answer(result, []):
            return result, doc_id
    return None, ""


async def decide(request, kit):
    invoice_candidates = unique_matches(INVOICE_PATTERNS, request, kit)
    doc_candidates = unique_matches(DOC_PATTERNS, request, kit)

    state = {
        "request": request,
        "invoice_candidates": invoice_candidates,
        "doc_candidates": doc_candidates,
    }

    questions = {
        "job": {
            "type": "choice",
            "instructions": (
                "Decide what the user is asking for by reading the whole request. "
                "Choose schedule_payment only if the user wants payment scheduled for an invoice already in the ledger, "
                "including requests to work out what is owed and then schedule it. "
                "Choose record_invoice only if the user wants a vendor invoice document entered into the ledger. "
                "Choose none if the request is about invoice status, cancellation, dispute, vendor information, "
                "or if it does not clearly ask to perform one of the two supported jobs. "
                "If the request mentions multiple items, choose the one the user explicitly asks to handle and ignore items "
                "described as needing nothing or being a different one."
            ),
            "criteria": {
                JOB_SCHEDULE: "The request asks to schedule payment for an invoice already in the ledger.",
                JOB_RECORD: "The request asks to enter a vendor invoice document into the ledger.",
                JOB_NONE: "The request is out of scope or does not clearly ask for one of the supported jobs.",
            },
        },
        "perform": {
            "type": "noul",
            "instructions": (
                "Is the user directly asking us to carry out the supported job now, rather than talking about status, "
                "cancellation, disputes, vendor details, or some other topic?"
            ),
        },
    }

    if len(invoice_candidates) > 0:
        invoice_criteria = {}
        for candidate in invoice_candidates:
            invoice_criteria[candidate] = "A candidate invoice identifier mentioned in the request."
        invoice_criteria[TARGET_NONE] = "None of the listed invoice candidates is the one to use."
        questions["invoice_target"] = {
            "type": "choice",
            "instructions": (
                "If the request is about scheduling a payment, pick the invoice candidate the user means. "
                "Choose the candidate the user wants handled, even if other invoice mentions are described as needing nothing "
                "or being a different invoice. Choose none if no candidate is actually the target."
            ),
            "criteria": invoice_criteria,
        }

    if len(doc_candidates) > 0:
        doc_criteria = {}
        for candidate in doc_candidates:
            doc_criteria[candidate] = "A candidate document identifier mentioned in the request."
        doc_criteria[TARGET_NONE] = "None of the listed document candidates is the one to use."
        questions["doc_target"] = {
            "type": "choice",
            "instructions": (
                "If the request is about entering a vendor invoice document, pick the document candidate the user means. "
                "Choose the candidate the user wants handled, even if other document mentions are described as needing nothing "
                "or being a different document. Choose none if no candidate is actually the target."
            ),
            "criteria": doc_criteria,
        }

    answers = await kit.ask(state, questions)

    if "job" not in answers or "perform" not in answers:
        return {"handoff": "could not classify the request"}

    job = answers["job"]["choice"]
    perform_prob = answers["perform"]["noul"]

    if job == JOB_NONE:
        return {"handoff": "request is not asking for a supported job"}

    if perform_prob < 0.7:
        return {"handoff": "request does not clearly ask to carry out the job now"}

    if job == JOB_SCHEDULE:
        if "invoice_target" not in answers:
            return {"handoff": "no invoice candidate found for payment scheduling"}

        target = answers["invoice_target"]["choice"]
        if target == TARGET_NONE:
            return {"handoff": "could not identify the invoice to schedule"}

        invoice_options = normalize_options(target, "invoice", kit)
        if len(invoice_options) == 0:
            return {"handoff": "could not identify the invoice to schedule"}

        payable, invoice_id = await try_amount_payable(invoice_options, kit)
        if not valid_answer(payable, ["invoice_id", "amount_usd", "pay_on"]):
            return {"handoff": "could not determine the payable amount for " + str(target)}

        amount = clean_value(payable["amount_usd"])
        pay_on = payable["pay_on"]
        return {
            "action": JOB_SCHEDULE,
            "arguments": {
                "invoice_id": invoice_id,
                "amount_usd": amount,
                "pay_on": pay_on,
            },
            "reply": "Scheduling payment for invoice " + str(invoice_id) + ".",
        }

    if job == JOB_RECORD:
        if "doc_target" not in answers:
            return {"handoff": "no document candidate found for invoice recording"}

        target = answers["doc_target"]["choice"]
        if target == TARGET_NONE:
            return {"handoff": "could not identify the document to record"}

        doc_options = normalize_options(target, "document", kit)
        if len(doc_options) == 0:
            return {"handoff": "could not identify the document to record"}

        parsed, doc_id = await try_parse_invoice(doc_options, kit)
        if not isinstance(parsed, dict):
            return {"handoff": "could not parse the document " + str(target)}

        record = clean_record(parsed)
        return {
            "action": JOB_RECORD,
            "arguments": record,
            "reply": "Recording document " + str(doc_id) + " in the ledger.",
        }

    return {"handoff": "request is not asking for a supported job"}
