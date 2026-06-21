"""Per-document-type extraction schemas + a registry keyed by DocumentType.

These define the "mandatory fields" extracted for each type. Every field is
Optional so missing data becomes ``null`` instead of breaking extraction, and
each model carries an ``extra_fields`` list to capture anything else printed on
the document (the assignment asks to extract *all* fields).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.common import (
    CustomsItem,
    ExtraField,
    InvoiceLineItem,
    PackingItem,
    Party,
)
from app.schemas.types import DocumentType


class InvoiceData(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    order_number: Optional[str] = Field(None, description="Purchase order / order reference.")
    seller: Optional[Party] = Field(None, description="Seller / exporter / 'from'.")
    buyer: Optional[Party] = Field(None, description="Buyer / importer / 'bill-to'.")
    ship_to: Optional[Party] = None
    incoterms: Optional[str] = Field(None, description="Delivery terms (EXW, CIP, DDP...).")
    payment_terms: Optional[str] = None
    currency: Optional[str] = None
    line_items: list[InvoiceLineItem] = Field(default_factory=list)
    subtotal: Optional[str] = None
    discount: Optional[str] = None
    tax_amount: Optional[str] = None
    total_amount: Optional[str] = Field(None, description="Grand total payable.")
    total_net_weight: Optional[str] = None
    total_gross_weight: Optional[str] = None
    country_of_origin: Optional[str] = None
    country_of_destination: Optional[str] = None
    bank_details: Optional[str] = None
    extra_fields: list[ExtraField] = Field(default_factory=list)


class AirWaybillData(BaseModel):
    awb_number: Optional[str] = Field(None, description="Air waybill number.")
    shipper: Optional[Party] = None
    consignee: Optional[Party] = None
    issuing_carrier_agent: Optional[Party] = None
    agent_iata_code: Optional[str] = None
    airport_of_departure: Optional[str] = None
    airport_of_destination: Optional[str] = None
    routing_and_destination: Optional[str] = None
    by_first_carrier: Optional[str] = None
    flight_number: Optional[str] = None
    flight_date: Optional[str] = None
    number_of_pieces: Optional[str] = None
    gross_weight: Optional[str] = None
    chargeable_weight: Optional[str] = None
    rate_class: Optional[str] = None
    rate_charge: Optional[str] = None
    nature_and_quantity_of_goods: Optional[str] = None
    hs_codes: list[str] = Field(default_factory=list)
    declared_value_for_carriage: Optional[str] = None
    declared_value_for_customs: Optional[str] = None
    currency: Optional[str] = None
    total_prepaid: Optional[str] = None
    total_collect: Optional[str] = None
    handling_information: Optional[str] = None
    volume: Optional[str] = None
    executed_date: Optional[str] = None
    executed_place: Optional[str] = None
    extra_fields: list[ExtraField] = Field(default_factory=list)


class CMRData(BaseModel):
    cmr_number: Optional[str] = None
    sender: Optional[Party] = Field(None, description="Sender (box 1).")
    consignee: Optional[Party] = Field(None, description="Consignee (box 2).")
    carrier: Optional[Party] = Field(None, description="Carrier (box 16).")
    successive_carriers: Optional[str] = Field(None, description="(box 17).")
    place_of_delivery: Optional[str] = Field(None, description="(box 3).")
    place_and_date_of_taking_over: Optional[str] = Field(None, description="(box 4).")
    documents_attached: Optional[str] = Field(None, description="(box 5).")
    goods_description: Optional[str] = Field(None, description="(boxes 6-9).")
    marks_and_numbers: Optional[str] = None
    number_of_packages: Optional[str] = None
    method_of_packing: Optional[str] = None
    gross_weight: Optional[str] = Field(None, description="(box 11).")
    volume: Optional[str] = Field(None, description="(box 12).")
    vehicle_registration: Optional[str] = None
    senders_instructions: Optional[str] = Field(None, description="(box 13).")
    charges: Optional[str] = Field(None, description="(box 20).")
    reservations_and_observations: Optional[str] = Field(None, description="(box 18).")
    special_agreements: Optional[str] = Field(None, description="(box 19).")
    established_place_and_date: Optional[str] = Field(None, description="(box 21).")
    extra_fields: list[ExtraField] = Field(default_factory=list)


class PackingListData(BaseModel):
    packing_list_number: Optional[str] = None
    date: Optional[str] = None
    seller: Optional[Party] = None
    buyer: Optional[Party] = None
    related_invoice_number: Optional[str] = None
    items: list[PackingItem] = Field(default_factory=list)
    total_packages: Optional[str] = None
    total_net_weight: Optional[str] = None
    total_gross_weight: Optional[str] = None
    total_volume: Optional[str] = None
    extra_fields: list[ExtraField] = Field(default_factory=list)


class CustomsDeclarationData(BaseModel):
    mrn: Optional[str] = Field(None, description="Movement Reference Number.")
    declaration_type: Optional[str] = Field(None, description="EX/IM + type, e.g. 'EX A' (box 1).")
    customs_office: Optional[str] = None
    exporter: Optional[Party] = Field(None, description="Exporter / consignor (box 2).")
    consignee: Optional[Party] = Field(None, description="Consignee (box 8).")
    declarant_representative: Optional[Party] = Field(None, description="(box 14).")
    country_of_dispatch: Optional[str] = Field(None, description="(box 15).")
    country_of_destination: Optional[str] = Field(None, description="(box 17).")
    country_of_origin: Optional[str] = None
    delivery_terms: Optional[str] = Field(None, description="Incoterms (box 20).")
    currency: Optional[str] = Field(None, description="Invoice currency (box 22).")
    total_invoice_amount: Optional[str] = Field(None, description="(box 22).")
    exchange_rate: Optional[str] = Field(None, description="(box 23).")
    total_gross_mass: Optional[str] = Field(None, description="(box 35).")
    total_net_mass: Optional[str] = None
    total_packages: Optional[str] = Field(None, description="(box 6).")
    transport_identity: Optional[str] = Field(None, description="Transport id at border (box 18/21).")
    transport_mode: Optional[str] = Field(None, description="(box 25).")
    items: list[CustomsItem] = Field(default_factory=list)
    documents_produced: Optional[str] = Field(None, description="(box 44).")
    total_customs_value: Optional[str] = None
    extra_fields: list[ExtraField] = Field(default_factory=list)


class LetterData(BaseModel):
    reference_number: Optional[str] = Field(None, description="Outgoing ref no. (Исх №).")
    date: Optional[str] = None
    sender: Optional[Party] = None
    recipient: Optional[str] = None
    subject: Optional[str] = None
    summary: Optional[str] = Field(None, description="2-3 sentence summary of the content.")
    signatory: Optional[str] = Field(None, description="Person who signed.")
    referenced_documents: list[str] = Field(
        default_factory=list, description="Any documents/numbers referenced in the letter."
    )
    extra_fields: list[ExtraField] = Field(default_factory=list)


class GenericDocumentData(BaseModel):
    """Best-effort extraction for unknown / 'other' documents."""

    title: Optional[str] = None
    summary: Optional[str] = None
    document_number: Optional[str] = None
    date: Optional[str] = None
    parties: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)
    key_values: list[ExtraField] = Field(default_factory=list)


SCHEMA_REGISTRY: dict[DocumentType, type[BaseModel]] = {
    DocumentType.INVOICE: InvoiceData,
    DocumentType.AIR_WAYBILL: AirWaybillData,
    DocumentType.CMR: CMRData,
    DocumentType.PACKING_LIST: PackingListData,
    DocumentType.CUSTOMS_DECLARATION: CustomsDeclarationData,
    DocumentType.LETTER: LetterData,
    DocumentType.OTHER: GenericDocumentData,
}


def get_schema(doc_type: DocumentType) -> type[BaseModel]:
    return SCHEMA_REGISTRY.get(doc_type, GenericDocumentData)
