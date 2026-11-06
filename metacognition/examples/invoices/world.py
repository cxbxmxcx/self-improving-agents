"""A simulated accounts-payable desk: vendors' invoices in their own layouts, purchase orders, three clients'
terms, a vendor directory, and an audit.

D3's world (DOMAINS.md). The desk does three kinds of work for three client companies:

- intake: read a vendor's invoice document, written in that vendor's own layout, and record it in the ledger;
- payment: work out what to pay on an entered invoice, and when, under the client's terms;
- questions: look up a vendor's remittance email (work no procedure covers).

Each vendor has its own layout: key/value lines (Kestrel), a ruled table (Marlow), a letter (Tern & Daughters)
and semicolon-separated rows (Quillon). Dates and terms are written differently in each.

Every client's terms document states the readable rules: pay only for what the purchase order shows as
delivered, add sales tax on the client's taxed categories (before any discount), take a vendor's early-payment
discount whenever it is offered (off the goods only, paying on the last discount day; otherwise pay on the net
due date), pay freight as invoiced, and round to the cent.

Each client also keeps one convention that is written nowhere. It shows only in the audit's verdicts:
- Alder & Finch Clinics' logistics team pays carriers itself, so freight is never part of an Alder payment;
- Bramblewood Schools' treasury pays whole dollars, rounding the payment down;
- Copperline Builders withholds 5% retention on construction materials until a project closes.

The drift: from request 721 on, Kestrel bills from its European entity. Its documents keep their labels but
write dates as DD.MM.YYYY and amounts with a decimal comma (1.234,50), so a parser written for the old layout
reads them wrongly without failing.

The audit accepts exactly one action that matches the work: the right ledger record for an intake, the right
amount and date for a payment, or an answer that names the vendor's remittance email. When it rejects, it says
what was wrong and names the rule most likely broken, without giving the right answer.
"""

from __future__ import annotations

import copy
import random
from datetime import date, timedelta
from typing import Any

TOLERANCE = 0.01
RETENTION = 0.05
DRIFT_FROM = 721  # Kestrel's European layout, from this request number on

CLIENTS = {
    "ALDER": {"name": "Alder & Finch Clinics", "sales_tax_rate": 0.0825, "taxed_categories": ["equipment"],
              "suppliers": ["KES", "QUI", "MAR"], "po_prefix": "ALD"},
    "BRAMBLE": {"name": "Bramblewood Schools", "sales_tax_rate": 0.06, "taxed_categories": ["equipment", "supplies"],
                "suppliers": ["MAR", "KES", "QUI"], "po_prefix": "BRM"},
    "COPPER": {"name": "Copperline Builders", "sales_tax_rate": 0.0725,
               "taxed_categories": ["equipment", "materials"], "suppliers": ["TERN", "KES"], "po_prefix": "COP"},
}
CLIENT_ORDER = ("ALDER", "BRAMBLE", "COPPER")
# Each client's convention, written nowhere: the rule key the audit uses for it.
TACIT = {"ALDER": "freight", "BRAMBLE": "whole_dollars", "COPPER": "retention"}

RULES = [
    "Pay only for the quantities the purchase order shows as delivered, never for more than was invoiced.",
    "Add sales tax at the client's rate on the delivered goods in the client's taxed categories, before any "
    "discount.",
    "When the vendor's terms offer an early-payment discount, take it: take the discount percent off the goods "
    "(not off tax or freight) and pay on the invoice date plus the discount days. Otherwise pay the full amount on "
    "the invoice date plus the net days.",
    "Freight is payable as invoiced.",
    "Round the payment to the cent.",
]

VENDORS = {
    "KES": {"name": "Kestrel Industrial Supply", "remittance_email": "remit@kestrel-supply.example",
            "terms": (2.0, 10, 30), "freight": 0.8,
            "items": [("KS-GLV", "Nitrile gloves, box of 100", "supplies", 9, 15),
                      ("KS-GLS", "Safety glasses", "supplies", 4, 8),
                      ("KS-TAP", "Duct tape, case of 24", "supplies", 30, 45),
                      ("KS-DRL", "Cordless drill", "equipment", 89, 149),
                      ("KS-LAD", "Step ladder", "equipment", 59, 99)]},
    "MAR": {"name": "Marlow Office Goods", "remittance_email": "payments@marlow-office.example",
            "terms": (0.0, 0, 45), "freight": 0.7,
            "items": [("MO-PPR", "Copy paper, case", "supplies", 28, 42),
                      ("MO-TNR", "Toner cartridge", "supplies", 55, 90),
                      ("MO-CHR", "Task chair", "equipment", 120, 240),
                      ("MO-SHR", "Paper shredder", "equipment", 80, 160)]},
    "TERN": {"name": "Tern & Daughters Timber", "remittance_email": "accounts@tern-timber.example",
             "terms": (1.5, 15, 30), "freight": 0.9,
             "items": [("TD-STD", "2x4 stud 8 ft", "materials", 3, 6),
                       ("TD-PLY", "plywood sheet 3/4 in", "materials", 32, 48),
                       ("TD-CEM", "cement bag 80 lb", "materials", 6, 10),
                       ("TD-SAW", "circular saw", "equipment", 99, 179)]},
    "QUI": {"name": "Quillon Medical", "remittance_email": "ar@quillon-medical.example",
            "terms": (1.0, 10, 30), "freight": 0.6,
            "items": [("QM-SYR", "Syringes 5 ml box of 100", "supplies", 14, 22),
                      ("QM-GZE", "Gauze pads box of 200", "supplies", 8, 14),
                      ("QM-MON", "Blood pressure monitor", "equipment", 45, 95),
                      ("QM-THR", "Digital thermometer", "equipment", 12, 25)]},
}
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")

KINDS = ("intake", "payment", "intake", "payment", "question", "payment", "intake", "payment")  # by number % 8
REQUESTS = {
    "intake": ("Record the vendor invoice in document {doc}.",
               "A vendor invoice came in as document {doc}: please enter it in the ledger.",
               "Enter document {doc} into the ledger; it is a vendor invoice."),
    "payment": ("Schedule the payment for invoice {inv}.",
                "Set up the payment of invoice {inv} under the client's terms.",
                "Work out what we owe on invoice {inv} and when, then schedule the payment."),
    "question": ("Which email should remittance advice for {vendor} go to?",
                 "What is the remittance email address for {vendor}?",
                 "{vendor} asked where we send remittance advice. Which email do they use for it?"),
}


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE + 1e-9


def _add(day: str, days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


# ------------------------------------------------------------------ the terms' arithmetic

def payment(task: dict[str, Any], *, ignore: str = "") -> tuple[float, str]:
    """What the client's terms give an entered invoice: (amount in dollars, the date to pay).

    `ignore` names one rule to break, the way a wrong payment most likely broke it, for diagnosis.
    """
    invoice, po = task["invoice"], task["po"]
    client = CLIENTS[invoice["client_id"]]
    delivered = {line["sku"]: line["delivered"] for line in po["lines"]}
    goods = taxed = materials = 0.0
    for line in invoice["lines"]:
        quantity = line["quantity"] if ignore == "delivered" else min(line["quantity"], delivered.get(line["sku"], 0))
        amount = quantity * line["unit_price"]
        goods += amount
        if ignore == "tax_all" or line["category"] in client["taxed_categories"]:
            taxed += amount
        if line["category"] == "materials":
            materials += amount
    terms = invoice["terms"]
    take = terms["discount_percent"] > 0 and ignore != "discount"
    tax = 0.0 if ignore == "tax" else taxed * client["sales_tax_rate"]
    freight = 0.0 if ignore == "freight_payable" else invoice["freight_usd"]
    base = goods + tax + freight if ignore == "discount_base" else goods
    discount = base * terms["discount_percent"] / 100 if take else 0.0
    total = goods - discount + tax + freight
    convention = TACIT[invoice["client_id"]]
    if convention == "freight" and ignore != "freight":
        total -= freight
    if convention == "retention" and ignore != "retention":
        total -= RETENTION * materials
    total = round(total, 2)
    if convention == "whole_dollars" and ignore != "whole_dollars":
        total = float(int(total))
    days = terms["discount_days"] if take and ignore != "pay_on" else terms["net_days"]
    return total, _add(invoice["invoice_date"], days)


# The mistakes a payment audit recognizes, most telling first. The first three are the clients' unwritten
# conventions, so these messages are the only place a learner can meet them.
PAYMENT_DIAGNOSES = {
    "freight": "it included freight, which Alder & Finch's logistics team pays to carriers itself",
    "whole_dollars": "it paid cents, but Bramblewood's treasury pays whole dollars, rounding down",
    "retention": "it paid in full for construction materials, on which Copperline withholds 5% retention",
    "delivered": "it paid for invoiced quantities the purchase order doesn't show as delivered",
    "tax": "it left out the sales tax on goods in the client's taxed categories",
    "tax_all": "it taxed goods in a category the client's terms don't tax",
    "discount": "it didn't take the vendor's early-payment discount",
    "discount_base": "it took the discount off tax and freight as well as the goods",
    "freight_payable": "it left out the freight, which is payable as invoiced",
    "pay_on": "it scheduled the payment for the wrong date under the vendor's terms",
}


def rules_that_matter(task: dict[str, Any]) -> set[str]:
    """The payment rules that change this invoice's payment."""
    expected = payment(task)
    out = set()
    for rule in PAYMENT_DIAGNOSES:
        broken = payment(task, ignore=rule)
        if not _same(broken[0], expected[0]) or broken[1] != expected[1]:
            out.add(rule)
    return out


def diagnose_payment(task: dict[str, Any], amount: float, pay_on: str) -> str:
    for rule, message in PAYMENT_DIAGNOSES.items():
        broken = payment(task, ignore=rule)
        if _same(broken[0], amount) and broken[1] == pay_on:
            return message
    expected = payment(task)
    if _same(expected[0], amount):
        return "the amount is right but the payment date isn't what the vendor's terms give"
    return "that is not what the client's terms give"


# ------------------------------------------------------------------ documents in each vendor's layout

def us(x: float) -> str:
    return f"{x:,.2f}"


def eu(x: float) -> str:
    return us(x).replace(",", "#").replace(".", ",").replace("#", ".")


def misread_eu(x: float) -> float:
    """What a parser for US amounts makes of `x` written the European way: commas dropped as separators."""
    return float(eu(x).replace(",", ""))


def subtotal(invoice: dict[str, Any]) -> float:
    return round(sum(line["quantity"] * line["unit_price"] for line in invoice["lines"]), 2)


def render(invoice: dict[str, Any], *, european: bool = False) -> str:
    """The invoice as its vendor writes it."""
    client = CLIENTS[invoice["client_id"]]
    day = date.fromisoformat(invoice["invoice_date"])
    pct, ddays, net = (invoice["terms"][k] for k in ("discount_percent", "discount_days", "net_days"))
    lines, freight = invoice["lines"], invoice["freight_usd"]
    if invoice["vendor_id"] == "KES":
        money = eu if european else us
        out = ["KESTREL INDUSTRIAL SUPPLY GMBH" if european else "KESTREL INDUSTRIAL SUPPLY",
               f"Invoice No: {invoice['invoice_number']}",
               f"Invoice Date: {day.strftime('%d.%m.%Y') if european else day.isoformat()}",
               f"Bill To: {client['name']}", f"PO Number: {invoice['po_number']}",
               f"Terms: {pct:g}/{ddays} net {net}", "", "SKU | Description | Qty | Unit"]
        out += [f"{li['sku']} | {li['description']} | {li['quantity']} | {money(li['unit_price'])}" for li in lines]
        out += ["", f"Subtotal: {money(subtotal(invoice))}"]
        if freight:
            out.append(f"Freight: {money(freight)}")
        out.append(f"Total Due: {money(subtotal(invoice) + freight)}")
        return "\n".join(out)
    if invoice["vendor_id"] == "MAR":
        rule = "+--------+------------------------------+-----+------------+------------+"
        out = ["MARLOW OFFICE GOODS", f"INVOICE {invoice['invoice_number']}",
               f"Dated {MONTHS[day.month - 1]} {day.day}, {day.year} | Customer PO {invoice['po_number']} | "
               f"Payment terms Net {net}", rule,
               "| Item   | Description                  | Qty | Unit price | Line total |", rule]
        out += [f"| {li['sku']} | {li['description']:<28} | {li['quantity']:>3} | {us(li['unit_price']):>10} | "
                f"{us(li['quantity'] * li['unit_price']):>10} |" for li in lines]
        out += [rule, f"Goods total {us(subtotal(invoice))} | Delivery charge {us(freight)} | "
                      f"Amount due {us(subtotal(invoice) + freight)}"]
        return "\n".join(out)
    if invoice["vendor_id"] == "TERN":
        supplied = "; ".join(f"{li['quantity']} x {li['description']} ({li['sku']}) at ${us(li['unit_price'])} each"
                             for li in lines)
        carriage = f"Carriage comes to ${us(freight)}." if freight else "There is no charge for carriage."
        return "\n".join([
            "Tern & Daughters Timber", "14 Mill Road, Harrow Creek", "", f"Dear {client['name']},", "",
            f"Thank you for your order {invoice['po_number']}. This letter is our invoice {invoice['invoice_number']}, "
            f"dated {day.day} {MONTHS[day.month - 1]} {day.year}.", "",
            f"We supplied: {supplied}.", "",
            f"{carriage} Please settle within {net} days; you may take {pct:g}% off the goods if you pay within "
            f"{ddays} days.", "", "Yours faithfully,", "Mara Tern"])
    out = [f"QUILLON MEDICAL;INVOICE;{invoice['invoice_number']}", f"DATE;{day.strftime('%m/%d/%Y')}",
           f"ORDER REF;{invoice['po_number']}", f"TERMS;{pct:g}/{ddays} N{net}", "ITEM;DESCRIPTION;QTY;PRICE"]
    out += [f"{li['sku']};{li['description']};{li['quantity']};{li['unit_price']:.2f}" for li in lines]
    if freight:
        out.append(f"SHIPPING;{freight:.2f}")
    return "\n".join(out)


def record_of(task: dict[str, Any]) -> dict[str, Any]:
    """The ledger record an intake should make: record_invoice's arguments."""
    invoice = task["invoice"]
    terms = invoice["terms"]
    return {"doc_id": task["doc_id"], "invoice_number": invoice["invoice_number"], "po_number": invoice["po_number"],
            "invoice_date": invoice["invoice_date"], "subtotal_usd": subtotal(invoice),
            "freight_usd": invoice["freight_usd"], "discount_percent": terms["discount_percent"],
            "discount_days": terms["discount_days"], "net_days": terms["net_days"]}


def diagnose_record(task: dict[str, Any], got: dict[str, Any]) -> list[str]:
    """What is wrong with a ledger record, field by field, without the right values."""
    want, problems = record_of(task), []
    european = task["drift"] and task["vendor"] == "KES"

    def number(key: str) -> float | None:
        try:
            return float(got.get(key))
        except (TypeError, ValueError):
            return None

    if str(got.get("doc_id", "")).strip().upper() != want["doc_id"]:
        problems.append("it recorded a different document")
    sub, freight = number("subtotal_usd"), number("freight_usd")
    if sub is None or not _same(sub, want["subtotal_usd"]):
        misread = sub is not None and (_same(sub, misread_eu(want["subtotal_usd"])) or sub > 5 * want["subtotal_usd"]
                                       or sub < want["subtotal_usd"] / 5)
        if european and misread:
            problems.append("it read the document's amounts in the wrong number format")
        elif sub is not None and _same(sub, want["subtotal_usd"] + want["freight_usd"]):
            problems.append("the subtotal includes the freight")
        else:
            problems.append("the subtotal doesn't match the document's lines")
    if freight is None or not _same(freight, want["freight_usd"]):
        if not problems or "number format" not in problems[-1]:
            problems.append("the freight doesn't match the document")
    when = str(got.get("invoice_date", "")).strip()
    if when != want["invoice_date"]:
        swapped = f"{want['invoice_date'][:4]}-{want['invoice_date'][8:10]}-{want['invoice_date'][5:7]}"
        if when == swapped:
            problems.append("it read the invoice date's day and month the wrong way round")
        elif len(when) != 10 or when[4:5] != "-" or when[7:8] != "-":
            problems.append("the invoice date isn't written as YYYY-MM-DD")
        else:
            problems.append("the invoice date doesn't match the document")
    for key, label in (("invoice_number", "invoice number"), ("po_number", "purchase-order number")):
        if str(got.get(key, "")).strip().upper() != str(want[key]).upper():
            problems.append(f"the {label} doesn't match the document")
    try:
        terms = (float(got.get("discount_percent")), int(got.get("discount_days")), int(got.get("net_days")))
    except (TypeError, ValueError):
        terms = None
    if terms is None or not (_same(terms[0], want["discount_percent"]) and terms[1:] == (want["discount_days"],
                                                                                           want["net_days"])):
        problems.append("the payment terms don't match the document")
    return problems


# ------------------------------------------------------------------ the world the tools see

class InvoiceWorld:
    def __init__(self, task: dict[str, Any]) -> None:
        self.task = task
        self.actions: list[tuple[str, dict[str, Any]]] = []

    def read_document(self, doc_id: str) -> dict[str, Any]:
        if self.task["kind"] != "intake" or doc_id.strip().upper() != self.task["doc_id"]:
            return {"error": f"no document {doc_id!r}"}
        return {"doc_id": self.task["doc_id"], "text": self.task["document"]}

    def read_invoice(self, invoice_id: str) -> dict[str, Any]:
        if self.task["kind"] != "payment" or invoice_id.strip().upper() != self.task["invoice"]["invoice_id"]:
            return {"error": f"no entered invoice {invoice_id!r}"}
        return copy.deepcopy(self.task["invoice"])

    def read_purchase_order(self, po_number: str) -> dict[str, Any]:
        if self.task["kind"] == "question" or po_number.strip().upper() != self.task["po"]["po_number"]:
            return {"error": f"no purchase order {po_number!r}"}
        return copy.deepcopy(self.task["po"])

    def read_client_terms(self, client_id: str) -> dict[str, Any]:
        key = client_id.strip().upper()
        found = next((cid for cid, c in CLIENTS.items() if key in (cid, c["name"].upper())), None)
        if found is None:
            return {"error": f"no client {client_id!r}"}
        client = CLIENTS[found]
        return {"client_id": found, "name": client["name"], "sales_tax_rate": client["sales_tax_rate"],
                "taxed_categories": list(client["taxed_categories"]), "rules": list(RULES)}

    def read_vendor(self, vendor: str) -> dict[str, Any]:
        key = vendor.strip().upper()
        found = next((vid for vid, v in VENDORS.items() if key == vid or key in v["name"].upper()), None)
        if found is None:
            return {"error": f"no vendor {vendor!r}"}
        v = VENDORS[found]
        return {"vendor_id": found, "name": v["name"], "remittance_email": v["remittance_email"],
                "phone": f"+1 555 01{list(VENDORS).index(found)}0", "terms": "see each invoice"}

    def record_invoice(self, **record: Any) -> dict[str, Any]:
        self.actions.append(("record_invoice", record))
        return {"status": "recorded", "doc_id": record.get("doc_id"), "ledger_id": f"LG-{len(self.actions)}"}

    def schedule_payment(self, invoice_id: str, amount_usd: float, pay_on: str) -> dict[str, Any]:
        self.actions.append(("schedule_payment",
                             {"invoice_id": invoice_id, "amount_usd": amount_usd, "pay_on": pay_on}))
        return {"status": "scheduled", "invoice_id": invoice_id, "amount_usd": round(float(amount_usd), 2),
                "pay_on": pay_on}

    def expected(self) -> Any:
        kind = self.task["kind"]
        if kind == "intake":
            return record_of(self.task)
        if kind == "payment":
            amount, pay_on = payment(self.task)
            return {"invoice_id": self.task["invoice"]["invoice_id"], "amount_usd": amount, "pay_on": pay_on}
        return {"remittance_email": VENDORS[self.task["vendor"]]["remittance_email"]}

    def check(self, answer: Any = None) -> dict[str, Any]:
        kind = self.task["kind"]
        if kind == "question":
            if self.actions:
                return {"success": False, "reason": "it acted on a question that needed only an answer"}
            email = VENDORS[self.task["vendor"]]["remittance_email"]
            if email in str(answer or "").lower():
                return {"success": True, "reason": "accepted"}
            return {"success": False, "reason": "the answer doesn't give the vendor's remittance email"}
        tool = "record_invoice" if kind == "intake" else "schedule_payment"
        if not self.actions:
            nothing = "nothing was recorded" if kind == "intake" else "no payment was scheduled"
            return {"success": False, "reason": nothing}
        if len(self.actions) > 1:
            return {"success": False, "reason": f"{len(self.actions)} actions were taken on one request"}
        name, args = self.actions[0]
        if name != tool:
            return {"success": False, "reason": f"it called {name} where the work needed {tool}"}
        if kind == "intake":
            problems = diagnose_record(self.task, args)
            if not problems:
                return {"success": True, "reason": "accepted"}
            return {"success": False, "reason": "; ".join(problems[:2])}
        if str(args.get("invoice_id", "")).strip().upper() != self.task["invoice"]["invoice_id"]:
            return {"success": False, "reason": "the payment was scheduled on a different invoice"}
        try:
            amount = float(args.get("amount_usd"))
        except (TypeError, ValueError):
            return {"success": False, "reason": "the payment has no amount"}
        pay_on = str(args.get("pay_on", "")).strip()
        expected = payment(self.task)
        if _same(amount, expected[0]) and pay_on == expected[1]:
            return {"success": True, "reason": "accepted"}
        return {"success": False, "reason": f"{amount:.2f} USD was scheduled for {pay_on}, but "
                                            f"{diagnose_payment(self.task, amount, pay_on)}"}


# ------------------------------------------------------------------ requests

def _lines(rng: random.Random, vendor: str) -> list[dict[str, Any]]:
    items = rng.sample(VENDORS[vendor]["items"], rng.randint(2, 3))
    return [{"sku": sku, "description": description, "category": category,
             "quantity": rng.choice((2, 3, 4, 5, 6, 8, 10, 12, 20, 24, 30, 40)),
             "unit_price": round(rng.randint(low * 100, high * 100 + 99) / 100, 2)}
            for sku, description, category, low, high in items]


def _invoice(rng: random.Random, number: int, kind: str, client_id: str, vendor: str, drift: bool) -> dict[str, Any]:
    client = CLIENTS[client_id]
    pct, ddays, net = VENDORS[vendor]["terms"]
    lines = _lines(rng, vendor)
    first = date(2026, 11, 2) if drift else date(2026, 8, 3)
    invoice_date = (first + timedelta(days=rng.randint(0, 45))).isoformat()
    charged = rng.random() < VENDORS[vendor]["freight"]
    freight = round(rng.randint(3, 19) * 5 + rng.choice((0.0, 0.5)), 2) if charged else 0.0
    number_text = {"KES": f"KIS-{40000 + number}", "MAR": f"MOG/2026/{number:04d}", "TERN": f"TDT-{9000 + number}",
                   "QUI": f"QM-{77000 + number}"}[vendor]
    po_number = f"PO-{client['po_prefix']}-{1000 + number}"
    invoice = {"invoice_id": f"INV-{number:03d}", "invoice_number": number_text, "vendor_id": vendor,
               "vendor_name": VENDORS[vendor]["name"], "client_id": client_id, "client_name": client["name"],
               "po_number": po_number, "invoice_date": invoice_date, "lines": lines, "freight_usd": freight,
               "terms": {"discount_percent": pct, "discount_days": ddays, "net_days": net}}
    po_lines = []
    for line in lines:
        short = rng.random() < (0.35 if kind == "payment" else 0.2)
        delivered = line["quantity"] - rng.randint(1, max(1, line["quantity"] // 2)) if short else line["quantity"]
        po_lines.append({"sku": line["sku"], "description": line["description"], "ordered": line["quantity"],
                         "delivered": delivered})
    po = {"po_number": po_number, "client_id": client_id, "vendor_id": vendor, "lines": po_lines}
    return {"invoice": invoice, "po": po}


def kind_of(number: int) -> str:
    if 801 <= number <= 806:
        return "intake"
    if 811 <= number <= 828:
        return "payment"
    return KINDS[number % 8]


def client_of(number: int) -> str:
    if 811 <= number <= 828:
        return CLIENT_ORDER[(number - 811) // 6]
    return CLIENT_ORDER[number % 3]


def hinges(number: int) -> bool:
    """Whether a payment must turn on its client's unwritten convention: three in four do."""
    return number >= 811 or (number // 2) % 4 != 0


def task(number: int) -> dict[str, Any]:
    """One request, from its number alone (ids V001..V828)."""
    rng = random.Random(number * 7919 + 17)
    kind, client_id = kind_of(number), client_of(number)
    drift = number >= DRIFT_FROM
    for _ in range(4000):
        if 801 <= number <= 806:
            vendor = ("KES", "KES", "KES", "MAR", "TERN", "QUI")[number - 801]
        elif kind == "intake" and drift and rng.random() < 0.6:
            vendor = "KES"
        else:
            vendor = rng.choice(CLIENTS[client_id]["suppliers"] if kind != "question" else sorted(VENDORS))
        case = {"number": number, "kind": kind, "client": client_id, "vendor": vendor, "drift": drift,
                **_invoice(rng, number, kind, client_id, vendor, drift)}
        if kind != "payment":
            break
        kinds = rules_that_matter(case)
        convention = TACIT[client_id] in kinds
        readable = kinds & {"delivered", "tax", "discount"}
        if convention == hinges(number) and readable:
            break
    else:
        raise RuntimeError(f"no request {number} meets the generator's conditions")
    phrasing = REQUESTS[kind][number % 3]
    if kind == "intake":
        case["doc_id"] = f"DOC-{number:03d}"
        case["document"] = render(case["invoice"], european=drift and vendor == "KES")
        text = phrasing.format(doc=case["doc_id"])
    elif kind == "payment":
        text = phrasing.format(inv=case["invoice"]["invoice_id"])
    else:
        text = phrasing.format(vendor=VENDORS[vendor]["name"])
    out = {"id": f"V{number:03d}", "text": text, **case}
    out["expected"] = InvoiceWorld(out).expected()
    return out


SPLITS = {"dev": range(1, 17), "validation": range(101, 117), "test": range(201, 217), "routing": range(301, 349),
          "stream": range(701, 761), "gate": range(801, 807), "payable_gate": range(811, 829)}


def task_set(split: str) -> list[dict[str, Any]]:
    return [task(n) for n in SPLITS[split]]


def task_by_id(task_id: str) -> dict[str, Any]:
    return task(int(task_id[1:]))
