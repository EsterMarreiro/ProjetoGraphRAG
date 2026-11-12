"""Texto corrido das normas e do portal da SUDEMA, dividido em trechos."""

from __future__ import annotations

import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pymupdf

CHUNK_LIMIT = 1400
NOISE_BLOCKS = re.compile(r"^\d{1,3}$")


def pdf_blocks(path: Path, first: int, last: int) -> list[tuple[int, str]]:
    document = pymupdf.open(path)
    blocks: list[tuple[int, str]] = []
    for number in range(first, last + 1):
        for block in document[number - 1].get_text("blocks", sort=True):
            text = re.sub(r"\s+", " ", block[4]).strip()
            if text and not NOISE_BLOCKS.match(text):
                blocks.append((number, text))
    return blocks


def chunk_blocks(
    blocks: list[tuple[int, str]], limit: int = CHUNK_LIMIT
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current = ""
    page = 0
    for number, text in blocks:
        pieces = (
            [text]
            if len(text) <= limit
            else [text[i : i + limit] for i in range(0, len(text), limit)]
        )
        for piece in pieces:
            if current and len(current) + len(piece) + 1 > limit:
                chunks.append({"page": page, "text": current})
                current, page = piece, number
            else:
                current = f"{current} {piece}".strip() if current else piece
                page = page or number
    if current:
        chunks.append({"page": page, "text": current})
    return chunks


class _Content(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.in_content = False
        self.skip = 0
        self.lines: list[str] = []
        self.buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id") == "content":
            self.in_content = True
            self.depth = 0
        if not self.in_content:
            return
        self.depth += 1 if tag in {"div", "section", "article"} else 0
        if tag in {"script", "style", "footer", "nav"}:
            self.skip += 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "br", "tr"}:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if not self.in_content:
            return
        if tag in {"script", "style", "footer", "nav"} and self.skip:
            self.skip -= 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "tr"}:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self.in_content and not self.skip:
            self.buffer.append(data)

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self.buffer)).strip()
        self.buffer = []
        if text.startswith("Utilizamos ferramentas e serviços de terceiros"):
            return
        if text and text not in {">", "Voltar ao topo"}:
            self.lines.append(text)


def site_text(url: str) -> list[str]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(request, timeout=30).read().decode("utf-8", "ignore")
    parser = _Content()
    parser.feed(html)
    parser._flush()
    return parser.lines
