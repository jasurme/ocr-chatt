"""Specialized per-type extraction prompts (the assignment's 'specialized prompt'
for each document type)."""

from __future__ import annotations

from app.schemas.types import DocumentType

_BASE_RULES = (
    "You are an expert customs/trade document data-extraction engine.\n"
    "Extract the requested fields into the structured schema.\n"
    "STRICT RULES:\n"
    "- Use ONLY information present in the document. If a field is absent, return null "
    "(empty list for lists). NEVER invent or guess values.\n"
    "- Keep values in their ORIGINAL language and script (do not translate).\n"
    "- Copy numbers, codes and identifiers exactly as printed.\n"
    "- For dates, prefer ISO format YYYY-MM-DD when the date is unambiguous; "
    "otherwise keep it as written.\n"
    "- Extract EVERY line item / commodity row you can see — do not stop early.\n"
    "- Put any clearly-labelled field that does not fit the schema into "
    "'extra_fields' as name/value pairs.\n"
)

_TYPE_GUIDANCE: dict[DocumentType, str] = {
    DocumentType.INVOICE: (
        "DOCUMENT: Commercial invoice.\n"
        "Capture seller/exporter and buyer/importer (name, address, country, VAT/tax id), "
        "invoice number & date, due date, purchase-order/order reference, Incoterms/delivery "
        "terms, payment terms, currency, every line item (item no, description, HS code, "
        "quantity, unit, unit price, line amount, batch number, expiry date, origin), "
        "subtotal, discount, tax, and the grand total. Also total net/gross weight and "
        "bank details if present."
    ),
    DocumentType.AIR_WAYBILL: (
        "DOCUMENT: Air Waybill (AWB).\n"
        "Capture the AWB number (e.g. 3 digit prefix + 8 digits), shipper and consignee "
        "(name/address/account), issuing carrier's agent and its IATA code, airport of "
        "departure and destination, routing/first carrier, flight number/date, number of "
        "pieces (RCP), gross weight, chargeable weight, rate class, rate/charge, nature & "
        "quantity of goods, any HS codes, declared value for carriage and for customs, "
        "currency, total prepaid/collect charges, handling information, volume, and the "
        "execution date/place."
    ),
    DocumentType.CMR: (
        "DOCUMENT: CMR international road consignment note (numbered boxes).\n"
        "Capture sender (box 1), consignee (box 2), place of delivery (box 3), place & date "
        "of taking over (box 4), documents attached (box 5), goods description/marks/packages "
        "(boxes 6-9), gross weight (box 11), volume (box 12), sender's instructions (box 13), "
        "carrier (box 16), successive carriers (box 17), reservations (box 18), special "
        "agreements (box 19), charges (box 20), place & date established (box 21), and the "
        "vehicle registration number."
    ),
    DocumentType.PACKING_LIST: (
        "DOCUMENT: Packing list.\n"
        "Capture packing list number & date, seller and buyer, the related invoice number, "
        "every packed item (description, HS code, quantity, unit, number of packages, package "
        "type, net weight, gross weight, dimensions, marks), and the totals (packages, net "
        "weight, gross weight, volume)."
    ),
    DocumentType.CUSTOMS_DECLARATION: (
        "DOCUMENT: Customs declaration (GTD / EU export or import declaration, EAD).\n"
        "Capture the MRN, declaration type (box 1, e.g. EX A), customs office, exporter "
        "(box 2), consignee (box 8), declarant/representative (box 14), country of dispatch "
        "(box 15) and destination (box 17), delivery terms/Incoterms (box 20), currency and "
        "total invoice amount (box 22), exchange rate (box 23), total gross mass (box 35) and "
        "net mass, total packages (box 6), transport identity & mode (boxes 18/21/25), "
        "documents produced (box 44), and EACH commodity item (item number, description "
        "box 31, HS/commodity code box 33, country of origin box 34, gross mass box 35, net "
        "mass box 38, customs value, statistical value box 46, procedure box 37, packages)."
    ),
    DocumentType.LETTER: (
        "DOCUMENT: Business letter / official correspondence.\n"
        "Capture the outgoing reference number (Исх №), date, sender (company/person), "
        "recipient, subject, a short summary of the content, the signatory, and any document "
        "numbers referenced in the body (e.g. AWB or invoice numbers)."
    ),
    DocumentType.OTHER: (
        "DOCUMENT: Unknown / other type.\n"
        "Do a best-effort extraction: a title, a short summary, any document number, date, "
        "parties mentioned, monetary amounts, and notable labelled key/value pairs."
    ),
}


def build_extraction_system_prompt(doc_type: DocumentType) -> str:
    guidance = _TYPE_GUIDANCE.get(doc_type, _TYPE_GUIDANCE[DocumentType.OTHER])
    return f"{_BASE_RULES}\n{guidance}"
