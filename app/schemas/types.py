"""Supported document types + human-readable descriptions for the classifier."""

from __future__ import annotations

from enum import Enum


class DocumentType(str, Enum):
    INVOICE = "invoice"
    AIR_WAYBILL = "air_waybill"
    CMR = "cmr"
    PACKING_LIST = "packing_list"
    CUSTOMS_DECLARATION = "customs_declaration"
    LETTER = "letter"
    OTHER = "other"

    @property
    def label(self) -> str:
        return _LABELS[self]


# Short descriptions injected into the classification prompt so the model knows
# exactly what each category means (in this customs/trade domain).
DOC_TYPE_DESCRIPTIONS: dict[DocumentType, str] = {
    DocumentType.INVOICE: (
        "Commercial/proforma invoice — a bill for goods sold between a seller "
        "and buyer; has invoice number, line items, prices and totals."
    ),
    DocumentType.AIR_WAYBILL: (
        "Air Waybill (AWB / авианакладная) — air cargo transport document with "
        "an 11-digit AWB number, shipper, consignee, airports and weights."
    ),
    DocumentType.CMR: (
        "CMR consignment note — international ROAD transport document (CMR "
        "convention) with sender, consignee, vehicle and delivery details."
    ),
    DocumentType.PACKING_LIST: (
        "Packing list — itemized list of packed goods with quantities, package "
        "counts, net/gross weights and dimensions (no prices)."
    ),
    DocumentType.CUSTOMS_DECLARATION: (
        "Customs declaration — GTD / ГТД / EU export or import declaration "
        "(EX/IM, EAD) with MRN, exporter/consignee, HS codes and customs value."
    ),
    DocumentType.LETTER: (
        "Business letter / official correspondence / cover note — free-form "
        "letter (e.g. from a company), not a structured trade document."
    ),
    DocumentType.OTHER: (
        "Anything that does not fit the categories above, or an unrecognizable / "
        "irrelevant document."
    ),
}

_LABELS: dict[DocumentType, str] = {
    DocumentType.INVOICE: "Invoice",
    DocumentType.AIR_WAYBILL: "Air Waybill",
    DocumentType.CMR: "CMR (Road Consignment Note)",
    DocumentType.PACKING_LIST: "Packing List",
    DocumentType.CUSTOMS_DECLARATION: "Customs Declaration (GTD)",
    DocumentType.LETTER: "Letter / Correspondence",
    DocumentType.OTHER: "Other / Unknown",
}


def classifier_type_menu() -> str:
    """A bulleted menu of types + descriptions for the prompt."""
    return "\n".join(
        f"- {dt.value}: {DOC_TYPE_DESCRIPTIONS[dt]}" for dt in DocumentType
    )
