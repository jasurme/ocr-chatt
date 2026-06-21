"""Turn an extracted-document result into CSV / Excel downloads.

Scalar fields (including nested party fields, flattened with dotted keys) become
a "Summary" table; repeating line-item lists become their own sheet/rows. The
CSV denormalizes line items with the header fields repeated per row — the format
a customs broker expects in a spreadsheet.
"""

from __future__ import annotations

import io

import pandas as pd


def flatten_fields(data: dict, prefix: str = "") -> dict:
    """Flatten nested dicts to dotted keys; join scalar lists; skip item lists."""
    out: dict[str, str] = {}
    for key, value in (data or {}).items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_fields(value, prefix=f"{full}."))
        elif isinstance(value, list):
            if value and all(isinstance(v, dict) for v in value):
                continue  # list of objects -> handled as an item table
            out[full] = "; ".join("" if v is None else str(v) for v in value)
        else:
            out[full] = "" if value is None else str(value)
    return out


def _item_tables(data: dict) -> dict[str, list[dict]]:
    return {
        k: v
        for k, v in (data or {}).items()
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
    }


def build_tables(result: dict) -> tuple[dict, dict[str, list[dict]]]:
    """Return (scalar_fields, item_tables) for a processed-document result."""
    extracted = result.get("extracted") or {}
    scalar = {
        "filename": result.get("filename"),
        "doc_type": result.get("doc_type"),
    }
    scalar.update(flatten_fields(extracted))
    return scalar, _item_tables(extracted)


def _sanitize_sheet(name: str) -> str:
    bad = '[]:*?/\\'
    clean = "".join(c for c in name if c not in bad)
    return (clean or "Sheet")[:31]


def to_csv_bytes(result: dict) -> bytes:
    """CSV: one row per line item with header fields repeated; else a single row."""
    scalar, tables = build_tables(result)
    if tables:
        # Use the first/primary item list (line_items or items).
        name = next(iter(tables))
        items_df = pd.DataFrame(tables[name])
        for col, val in reversed(list(scalar.items())):
            items_df.insert(0, col, val)  # header fields as leading columns
        df = items_df
    else:
        df = pd.DataFrame([scalar])
    return df.to_csv(index=False).encode("utf-8-sig")  # BOM => Excel opens UTF-8 cleanly


def to_excel_bytes(result: dict) -> bytes:
    """Excel: a 'Summary' sheet of fields + one sheet per item list."""
    scalar, tables = build_tables(result)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary = pd.DataFrame(
            [{"Field": k, "Value": v} for k, v in scalar.items()]
        )
        summary.to_excel(writer, sheet_name="Summary", index=False)
        for name, rows in tables.items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=_sanitize_sheet(name), index=False)
    return buffer.getvalue()
