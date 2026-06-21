"""Document type taxonomy + structured extraction schemas."""

from app.schemas.common import (
    CustomsItem,
    ExtraField,
    InvoiceLineItem,
    PackingItem,
    Party,
)
from app.schemas.documents import (
    SCHEMA_REGISTRY,
    AirWaybillData,
    CMRData,
    CustomsDeclarationData,
    GenericDocumentData,
    InvoiceData,
    LetterData,
    PackingListData,
    get_schema,
)
from app.schemas.types import (
    DOC_TYPE_DESCRIPTIONS,
    DocumentType,
    classifier_type_menu,
)

__all__ = [
    "DocumentType",
    "DOC_TYPE_DESCRIPTIONS",
    "classifier_type_menu",
    "Party",
    "ExtraField",
    "InvoiceLineItem",
    "PackingItem",
    "CustomsItem",
    "InvoiceData",
    "AirWaybillData",
    "CMRData",
    "PackingListData",
    "CustomsDeclarationData",
    "LetterData",
    "GenericDocumentData",
    "SCHEMA_REGISTRY",
    "get_schema",
]
