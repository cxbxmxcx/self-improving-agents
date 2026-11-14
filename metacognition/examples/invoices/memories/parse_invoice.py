def parse_invoice(api, doc_id: str) -> dict:
    """The hand-written parser: every vendor's layout, Kestrel's European one included."""
    months = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7,
              "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}

    def money(text, comma_decimal):
        text = text.strip().replace("$", "").replace(" ", "")
        if comma_decimal:
            return float(text.replace(".", "").replace(",", "."))
        return float(text.replace(",", ""))

    def iso(year, month, day):
        return str(int(year)).zfill(4) + "-" + str(int(month)).zfill(2) + "-" + str(int(day)).zfill(2)

    def terms_of(text):
        # "2/10 net 30" or "1/10 N30": the discount percent, the discount days, the net days
        first, rest = text.strip().split(" ", 1)
        percent, days = first.split("/")
        return float(percent), int(days), int(rest.lower().replace("net", "").replace("n", "").strip())

    text = api.read_document(doc_id)["text"]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    record = {"doc_id": doc_id, "freight_usd": 0.0, "discount_percent": 0.0, "discount_days": 0}
    goods = 0.0

    if lines[0].startswith("KESTREL"):
        european = lines[0].endswith("GMBH")
        for line in lines:
            if line.startswith("Invoice No:"):
                record["invoice_number"] = line.split(":", 1)[1].strip()
            elif line.startswith("Invoice Date:"):
                value = line.split(":", 1)[1].strip()
                if european:
                    day, month, year = value.split(".")
                    value = iso(year, month, day)
                record["invoice_date"] = value
            elif line.startswith("PO Number:"):
                record["po_number"] = line.split(":", 1)[1].strip()
            elif line.startswith("Terms:"):
                percent, days, net = terms_of(line.split(":", 1)[1])
                record["discount_percent"], record["discount_days"], record["net_days"] = percent, days, net
            elif line.startswith("Freight:"):
                record["freight_usd"] = money(line.split(":", 1)[1], european)
            elif line.count("|") == 3 and not line.startswith("SKU"):
                cells = [cell.strip() for cell in line.split("|")]
                goods += int(cells[2]) * money(cells[3], european)

    elif lines[0].startswith("MARLOW"):
        for line in lines:
            if line.startswith("INVOICE "):
                record["invoice_number"] = line.split(" ", 1)[1].strip()
            elif line.startswith("Dated "):
                parts = [part.strip() for part in line.split("|")]
                month_name, day, year = parts[0][len("Dated "):].replace(",", "").split(" ")
                record["invoice_date"] = iso(year, months[month_name], day)
                record["po_number"] = parts[1][len("Customer PO "):].strip()
                record["net_days"] = int(parts[2].split("Net", 1)[1].strip())
            elif line.startswith("| ") and not line.startswith("| Item"):
                cells = [cell.strip() for cell in line.split("|")]
                goods += int(cells[3]) * money(cells[4], False)
            elif line.startswith("Goods total"):
                for part in line.split("|"):
                    if part.strip().startswith("Delivery charge"):
                        record["freight_usd"] = money(part.strip()[len("Delivery charge"):], False)

    elif lines[0].startswith("Tern"):
        for line in lines:
            if line.startswith("Thank you for your order"):
                record["po_number"] = line.split("order ", 1)[1].split(".", 1)[0].strip()
                record["invoice_number"] = line.split("our invoice ", 1)[1].split(",", 1)[0].strip()
                day, month_name, year = line.split("dated ", 1)[1].rstrip(".").split(" ")
                record["invoice_date"] = iso(year, months[month_name], day)
            elif line.startswith("We supplied:"):
                for item in line[len("We supplied:"):].strip().rstrip(".").split("; "):
                    quantity = int(item.split(" x ", 1)[0])
                    goods += quantity * money(item.split(" at $", 1)[1].split(" each", 1)[0], False)
            elif "settle within" in line:
                if "Carriage comes to $" in line:
                    record["freight_usd"] = money(line.split("Carriage comes to $", 1)[1].split(". ", 1)[0], False)
                record["net_days"] = int(line.split("settle within ", 1)[1].split(" days", 1)[0])
                record["discount_percent"] = float(line.split("take ", 1)[1].split("%", 1)[0])
                record["discount_days"] = int(line.split("if you pay within ", 1)[1].split(" days", 1)[0])

    elif lines[0].startswith("QUILLON"):
        for line in lines:
            cells = [cell.strip() for cell in line.split(";")]
            if cells[0] == "QUILLON MEDICAL":
                record["invoice_number"] = cells[2]
            elif cells[0] == "DATE":
                month, day, year = cells[1].split("/")
                record["invoice_date"] = iso(year, month, day)
            elif cells[0] == "ORDER REF":
                record["po_number"] = cells[1]
            elif cells[0] == "TERMS":
                percent, days, net = terms_of(cells[1])
                record["discount_percent"], record["discount_days"], record["net_days"] = percent, days, net
            elif cells[0] == "SHIPPING":
                record["freight_usd"] = money(cells[1], False)
            elif len(cells) == 4 and cells[0] != "ITEM":
                goods += int(cells[2]) * money(cells[3], False)

    record["subtotal_usd"] = round(goods, 2)
    return record
