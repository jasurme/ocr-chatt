"""Scraper + parser for lex.uz legislation (e.g. the Customs Code of Uzbekistan).

lex.uz serves the full legal text as static HTML (no JS needed). Each clause is
a `.lx_elem` block; per-clause UI toolbars are `<span onclick=...>` and
editorial notes are `.lx_no_select` — both are stripped before parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124 Safari/537.36"
)

# Official Customs Code of Uzbekistan, per language version on lex.uz.
# (marker captures the article number.)
LEXUZ_SOURCES: dict[str, dict] = {
    "uz": {
        # \b so the lookahead matches only at the start of a number
        # ("10-modda", not also "0-modda" inside it).
        "url": "https://lex.uz/docs/-2876354",
        "marker": r"\b(\d+)-modda\.",
        "title": "Oʻzbekiston Respublikasining Bojxona kodeksi",
    },
    "ru": {
        "url": "https://lex.uz/docs/2876352",
        "marker": r"Стать[яи]\s+(\d+)\.",
        "title": "Таможенный кодекс Республики Узбекистан",
    },
    "en": {
        "url": "https://lex.uz/docs/5535133",
        "marker": r"Article\s+(\d+)\.",
        "title": "Customs Code of the Republic of Uzbekistan",
    },
}


@dataclass
class Article:
    number: int
    title: str  # the article heading, e.g. "34-modda. Tovarni reeksport ..."
    body: str  # the article text without the heading line
    source_url: str
    language: str
    document_title: str = ""

    @property
    def text(self) -> str:
        """Full article: heading line + body (the embedded representation)."""
        return f"{self.title}\n{self.body}".strip()


def fetch_html(url: str, timeout: float = 60.0) -> str:
    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _clean_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for el in soup.select("[onclick]"):  # per-clause UI action toolbars
        el.decompose()
    for el in soup.select(".lx_no_select"):  # editorial notes / change history
        el.decompose()
    return soup


def _content_text(soup: BeautifulSoup) -> str:
    blocks: list[str] = []
    for el in soup.select(".lx_elem"):
        if el.find_parent(class_="lx_elem"):
            continue  # keep only top-level blocks (avoid nested duplication)
        if el.find_parent(class_=re.compile("docNavbar|docNabvar")):
            continue
        txt = el.get_text(" ", strip=True)
        if txt:
            blocks.append(re.sub(r"\s+", " ", txt).strip())
    return "\n".join(blocks)


def parse_articles(
    html: str,
    language: str = "uz",
    source_url: str = "",
    marker: str | None = None,
    document_title: str = "",
) -> list[Article]:
    """Parse legislation HTML into a list of :class:`Article`.

    Each article keeps its full heading line as ``title`` (e.g.
    "34-modda. Tovarni reeksport ...") and the remaining text as ``body``.
    """
    marker = marker or LEXUZ_SOURCES.get(language, LEXUZ_SOURCES["uz"])["marker"]
    full = _content_text(_clean_soup(html))

    article_re = re.compile(marker)
    # Split keeping the marker at the start of each piece.
    pieces = re.split(rf"(?={marker})", full)
    articles: list[Article] = []
    for piece in pieces:
        piece = piece.strip()
        m = article_re.match(piece)
        if not m:
            continue
        number = int(m.group(1))
        split = piece.split("\n", 1)
        heading = split[0].strip()  # "34-modda. Tovarni reeksport ..."
        body = split[1].strip() if len(split) > 1 else ""
        articles.append(
            Article(
                number=number,
                title=heading,
                body=body,
                source_url=source_url,
                language=language,
                document_title=document_title,
            )
        )
    return articles


def scrape_customs_code(language: str = "uz", limit: int | None = None) -> list[Article]:
    """Fetch + parse the Customs Code for a language ('uz' | 'ru' | 'en')."""
    src = LEXUZ_SOURCES.get(language)
    if not src:
        raise ValueError(f"No lex.uz source configured for language {language!r}")
    html = fetch_html(src["url"])
    articles = parse_articles(
        html,
        language=language,
        source_url=src["url"],
        marker=src["marker"],
        document_title=src["title"],
    )
    return articles[:limit] if limit else articles
