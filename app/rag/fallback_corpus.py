"""A small curated knowledge base used when live lex.uz scraping is unavailable
(offline demos, CI). Concise summaries of Uzbekistan customs concepts in Russian
and English so the RAG mode always has something factual to retrieve.
"""

from __future__ import annotations

from langchain_core.documents import Document

_SOURCE = "https://lex.uz/docs/2876352"
_DOC = "Customs Code of the Republic of Uzbekistan (curated summary)"

_ENTRIES = [
    {
        "article": 1,
        "title": "Scope of the Customs Code",
        "lang": "en",
        "text": (
            "The Customs Code of the Republic of Uzbekistan governs the movement of "
            "goods and vehicles across the customs border, the levying of customs "
            "payments, customs clearance (registration), customs control, and the "
            "prevention and detection of customs offences."
        ),
    },
    {
        "article": 5,
        "title": "Customs territory and customs border",
        "lang": "en",
        "text": (
            "The customs territory of Uzbekistan comprises its land territory, "
            "territorial and internal waters and the airspace above them. The limits "
            "of the customs territory, and free customs zones and free warehouses, "
            "form the customs border of the Republic of Uzbekistan."
        ),
    },
    {
        "article": 0,
        "title": "Customs payments",
        "lang": "en",
        "text": (
            "Customs payments include the customs duty, value added tax (VAT) and "
            "excise tax levied on imported goods, and customs fees (for clearance, "
            "storage and customs escort). They must be paid before or at the time of "
            "submitting the customs declaration."
        ),
    },
    {
        "article": 0,
        "title": "Таможенные режимы (customs regimes)",
        "lang": "ru",
        "text": (
            "Таможенный кодекс предусматривает различные таможенные режимы, в том "
            "числе выпуск для свободного обращения (импорт), экспорт, таможенный "
            "транзит, таможенный склад, временный ввоз и вывоз, переработку на "
            "таможенной территории и реэкспорт. Режим определяет порядок уплаты "
            "таможенных платежей и применения мер контроля."
        ),
    },
    {
        "article": 0,
        "title": "Таможенная декларация (customs declaration)",
        "lang": "ru",
        "text": (
            "Товары, перемещаемые через таможенную границу, подлежат декларированию "
            "таможенному органу. Грузовая таможенная декларация (ГТД) содержит "
            "сведения о товарах, их таможенной стоимости, коде ТН ВЭД, стране "
            "происхождения и назначения, а также о применяемом таможенном режиме."
        ),
    },
    {
        "article": 0,
        "title": "Customs value and HS classification",
        "lang": "en",
        "text": (
            "The customs value of imported goods is normally the transaction value — "
            "the price actually paid or payable — adjusted for certain costs such as "
            "transport and insurance. Goods are classified under the Harmonized "
            "System (HS / ТН ВЭД) commodity nomenclature, which determines the "
            "applicable duty rate."
        ),
    },
]


def fallback_documents() -> list[Document]:
    docs: list[Document] = []
    for e in _ENTRIES:
        meta = {
            "source_url": _SOURCE,
            "document_title": _DOC,
            "article_number": e["article"],
            "title": e["title"],
            "language": e["lang"],
            "origin": "fallback",
        }
        docs.append(Document(page_content=f"{e['title']}\n{e['text']}", metadata=meta))
    return docs
