"""Tests for CSV / Excel export (Step 10). Fully offline."""

from __future__ import annotations

import csv
import io

import pandas as pd

from app.export import build_tables, flatten_fields, to_csv_bytes, to_excel_bytes

RESULT = {
    "filename": "инв.PDF",
    "doc_type": "invoice",
    "extracted": {
        "invoice_number": "1090130561",
        "currency": "EUR",
        "seller": {"name": "ORAFOL Europe GmbH", "country": "Germany"},
        "hs_codes": ["9027899000", "3822190090"],
        "line_items": [
            {"description": "641G White", "quantity": "300", "unit": "M2"},
            {"description": "641M Deep sea", "quantity": "50", "unit": "M2"},
        ],
        "notes": None,
    },
}

EMPTY_RESULT = {"filename": "x.pdf", "doc_type": "other", "extracted": {}}


# ----------------------------- flatten ------------------------------------- #
def test_flatten_nested_and_lists():
    flat = flatten_fields(RESULT["extracted"])
    assert flat["invoice_number"] == "1090130561"
    assert flat["seller.name"] == "ORAFOL Europe GmbH"
    assert flat["hs_codes"] == "9027899000; 3822190090"
    assert flat["notes"] == ""  # None -> empty string
    assert "line_items" not in flat  # object list excluded from scalars


def test_build_tables_separates_items():
    scalar, tables = build_tables(RESULT)
    assert scalar["doc_type"] == "invoice"
    assert "line_items" in tables and len(tables["line_items"]) == 2


# ------------------------------- CSV --------------------------------------- #
def test_csv_denormalizes_line_items():
    data = to_csv_bytes(RESULT).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(data)))
    assert len(rows) == 2  # one row per line item
    assert rows[0]["invoice_number"] == "1090130561"  # header repeated
    assert rows[0]["description"] == "641G White"
    assert rows[1]["quantity"] == "50"


def test_csv_without_items_single_row():
    data = to_csv_bytes(EMPTY_RESULT).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(data)))
    assert len(rows) == 1
    assert rows[0]["doc_type"] == "other"


# ------------------------------- Excel ------------------------------------- #
def test_excel_has_summary_and_item_sheets():
    raw = to_excel_bytes(RESULT)
    assert raw[:2] == b"PK"  # xlsx is a zip
    xls = pd.ExcelFile(io.BytesIO(raw))
    assert "Summary" in xls.sheet_names
    assert "line_items" in xls.sheet_names
    summary = pd.read_excel(xls, "Summary")
    assert "invoice_number" in set(summary["Field"])
    items = pd.read_excel(xls, "line_items")
    assert len(items) == 2 and "description" in items.columns


def test_excel_empty_has_summary_only():
    xls = pd.ExcelFile(io.BytesIO(to_excel_bytes(EMPTY_RESULT)))
    assert xls.sheet_names == ["Summary"]
