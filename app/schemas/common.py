"""Reusable sub-models shared across document schemas.

All fields are Optional: real documents are inconsistent, and the extractor must
return `null` for anything that is genuinely absent rather than hallucinate.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ExtraField(BaseModel):
    """A labelled field that appears on the document but isn't in the schema.

    Lets us honor the "extract ALL fields" requirement without an open-ended
    object (which strict structured-output schemas disallow).
    """

    name: str = Field(description="The field label as printed on the document.")
    value: str = Field(description="The field's value, verbatim.")


class Party(BaseModel):
    """A company/person involved in the transaction (seller, consignee, etc.)."""

    name: Optional[str] = Field(None, description="Legal/company or person name.")
    address: Optional[str] = Field(None, description="Full postal address (one string).")
    country: Optional[str] = Field(None, description="Country.")
    tax_id: Optional[str] = Field(None, description="VAT / INN / tax registration number.")
    contact: Optional[str] = Field(None, description="Phone / email / contact person.")
    account_number: Optional[str] = Field(None, description="Account number if shown.")


class InvoiceLineItem(BaseModel):
    item_no: Optional[str] = None
    description: Optional[str] = None
    hs_code: Optional[str] = Field(None, description="HS / commodity / tariff code.")
    quantity: Optional[str] = None
    unit: Optional[str] = Field(None, description="Unit of measure (pcs, kg, m2, ROL...).")
    unit_price: Optional[str] = None
    amount: Optional[str] = Field(None, description="Line total.")
    batch_number: Optional[str] = None
    expiry_date: Optional[str] = None
    country_of_origin: Optional[str] = None


class PackingItem(BaseModel):
    description: Optional[str] = None
    hs_code: Optional[str] = None
    quantity: Optional[str] = None
    unit: Optional[str] = None
    number_of_packages: Optional[str] = None
    package_type: Optional[str] = Field(None, description="Carton, pallet, box, ...")
    net_weight: Optional[str] = None
    gross_weight: Optional[str] = None
    dimensions: Optional[str] = None
    marks_and_numbers: Optional[str] = None


class CustomsItem(BaseModel):
    """One commodity line ('Dec. G. I.' item) of a customs declaration."""

    item_number: Optional[str] = None
    description: Optional[str] = Field(None, description="Description of goods (box 31).")
    commodity_code_hs: Optional[str] = Field(None, description="HS/CN code (box 33).")
    country_of_origin: Optional[str] = Field(None, description="(box 34).")
    gross_mass: Optional[str] = Field(None, description="Gross mass kg (box 35).")
    net_mass: Optional[str] = Field(None, description="Net mass kg (box 38).")
    customs_value: Optional[str] = None
    statistical_value: Optional[str] = Field(None, description="(box 46).")
    procedure: Optional[str] = Field(None, description="Customs procedure code (box 37).")
    number_of_packages: Optional[str] = None
    supplementary_units: Optional[str] = None
