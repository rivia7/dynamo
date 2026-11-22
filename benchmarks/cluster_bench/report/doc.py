# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tiny document model rendered to Markdown and to a single-file HTML page.

Sections are built once as a list of blocks; both renderers walk the same
list so the two report formats always carry the same numbers. Inline text is
a tuple of styled spans, so data values never need ad-hoc escaping by the
section builders: each renderer escapes every span for its own syntax.

The HTML renderer emits no scripts, no external URLs and no web fonts. Text
that contains ``://`` (for example an etcd endpoint in an env listing) is
written with entity-encoded slashes, so the page displays it verbatim while
the markup itself contains no URL.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Callable, Literal, Sequence, Union

Style = Literal["plain", "strong", "code", "em", "raw"]  # raw: Markdown as is
Tone = Literal["normal", "note", "warn"]

SPDX_MD_HEADER = (
    "<!--\n"
    "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved.\n"
    "SPDX-License-Identifier: Apache-2.0\n"
    "-->\n"
)


@dataclass(frozen=True)
class Span:
    """A run of text with one inline style."""

    text: str
    style: Style = "plain"


Inline = tuple[Span, ...]
Cell = Union[str, Span, Inline]


def strong(text: object) -> Span:
    """Bold span."""
    return Span(str(text), "strong")


def code(text: object) -> Span:
    """Inline-code span."""
    return Span(str(text), "code")


def em(text: object) -> Span:
    """Emphasis span."""
    return Span(str(text), "em")


def raw(text: object) -> Span:
    """Text that is already Markdown (rendered verbatim; escaped in HTML)."""
    return Span(str(text), "raw")


def inline(*parts: Cell) -> Inline:
    """Flatten strings, spans and inline tuples into one inline tuple."""
    spans: list[Span] = []
    for part in parts:
        if isinstance(part, Span):
            spans.append(part)
        elif isinstance(part, tuple):
            spans.extend(part)
        elif part:
            spans.append(Span(part))
    return tuple(spans)


def plain_text(content: Cell) -> str:
    """The text of a cell without any styling."""
    return "".join(span.text for span in inline(content))


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Heading:
    """Section heading; level-2 headings with an anchor feed the TOC."""

    text: str
    level: int = 2
    anchor: str = ""


@dataclass(frozen=True)
class Paragraph:
    """A paragraph; ``note``/``warn`` render as callouts."""

    content: Inline
    tone: Tone = "normal"


@dataclass(frozen=True)
class Bullets:
    """An unordered list."""

    items: tuple[Inline, ...]


@dataclass(frozen=True)
class Table:
    """A table; ``numeric`` columns are right-aligned."""

    headers: tuple[str, ...]
    rows: tuple[tuple[Cell, ...], ...]
    numeric: tuple[int, ...] = ()  # column indexes that are right-aligned numbers
    caption: str = ""


@dataclass(frozen=True)
class Figure:
    """An inline SVG chart with its caption."""

    chart_id: str
    svg: str
    caption: str


@dataclass(frozen=True)
class Code:
    """A preformatted code block."""

    text: str
    lang: str = ""


@dataclass(frozen=True)
class Kpi:
    """One headline number (stat tile)."""

    label: str
    value: str
    detail: str = ""
    unit: str = ""


@dataclass(frozen=True)
class KpiRow:
    """A row of headline numbers."""

    tiles: tuple[Kpi, ...]
    headers: tuple[str, str, str] = ("", "", "")  # Markdown table header


Block = Union[Heading, Paragraph, Bullets, Table, Figure, Code, KpiRow]


@dataclass(frozen=True)
class Document:
    """A titled list of blocks, rendered by both renderers."""

    lang: str
    title: str
    subtitle: str
    blocks: tuple[Block, ...]
    toc_label: str = ""
    footer: str = ""
    figure_label: str = "{n}: {caption}"  # str.format with ``n`` and ``caption``

    @property
    def figures(self) -> tuple[Figure, ...]:
        return tuple(b for b in self.blocks if isinstance(b, Figure))

    def figure_number(self, chart_id: str) -> int:
        for index, figure in enumerate(self.figures, start=1):
            if figure.chart_id == chart_id:
                return index
        raise KeyError(chart_id)

    def figure_caption(self, figure: Figure) -> str:
        return self.figure_label.format(
            n=self.figure_number(figure.chart_id), caption=figure.caption
        )


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>])")


def md_escape(text: str, *, table: bool = False) -> str:
    """Escape Markdown inline syntax (and pipes/newlines inside tables)."""
    out = _MD_SPECIAL.sub(r"\\\1", text)
    if table:
        out = out.replace("|", r"\|").replace("\n", " ")
    return out


def _md_code_span(text: str, *, table: bool) -> str:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    body = text.replace("|", r"\|").replace("\n", " ") if table else text
    return f"{fence}{pad}{body}{pad}{fence}"


def md_inline(content: Cell, *, table: bool = False) -> str:
    """Markdown for inline content (``table`` also escapes pipes)."""
    out: list[str] = []
    for span in inline(content):
        if not span.text:
            continue
        if span.style == "code":
            out.append(_md_code_span(span.text, table=table))
        elif span.style == "raw":
            text = span.text
            out.append(text.replace("|", r"\|").replace("\n", " ") if table else text)
        elif span.style == "strong":
            out.append(f"**{md_escape(span.text, table=table)}**")
        elif span.style == "em":
            out.append(f"*{md_escape(span.text, table=table)}*")
        else:
            out.append(md_escape(span.text, table=table))
    return "".join(out)


def md_table(
    headers: Sequence[Cell], rows: Sequence[Sequence[Cell]], numeric: Sequence[int] = ()
) -> str:
    """A GitHub-flavored Markdown table."""
    width = len(headers)
    lines = ["| " + " | ".join(md_inline(h, table=True) for h in headers) + " |"]
    lines.append(
        "|" + "|".join("---:" if i in numeric else "---" for i in range(width)) + "|"
    )
    for row in rows:
        cells = [md_inline(c, table=True) for c in row]
        cells += [""] * (width - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _md_fence(text: str) -> str:
    longest = max((len(run) for run in re.findall(r"`{3,}", text)), default=0)
    return "`" * max(3, longest + 1)


def render_markdown(doc: Document, figure_href: Callable[[str], str]) -> str:
    """Render ``doc`` as Markdown; figures link to ``figure_href(chart_id)``."""
    parts: list[str] = [SPDX_MD_HEADER, f"# {md_escape(doc.title)}", ""]
    if doc.subtitle:
        parts += [md_escape(doc.subtitle), ""]
    for block in doc.blocks:
        parts.append(_md_block(doc, block, figure_href))
        parts.append("")
    if doc.footer:
        parts += ["---", "", md_escape(doc.footer), ""]
    return "\n".join(parts).rstrip("\n") + "\n"


def _md_block(doc: Document, block: Block, figure_href: Callable[[str], str]) -> str:
    if isinstance(block, Heading):
        return f"{'#' * block.level} {md_escape(block.text)}"
    if isinstance(block, Paragraph):
        text = md_inline(block.content)
        return f"> {text}" if block.tone != "normal" else text
    if isinstance(block, Bullets):
        return "\n".join(f"- {md_inline(item)}" for item in block.items)
    if isinstance(block, Table):
        table = md_table(block.headers, block.rows, block.numeric)
        return f"{md_escape(block.caption)}\n\n{table}" if block.caption else table
    if isinstance(block, Figure):
        caption = md_escape(doc.figure_caption(block))
        alt = md_escape(block.caption)
        return f"![{alt}]({figure_href(block.chart_id)})\n\n*{caption}*"
    if isinstance(block, Code):
        fence = _md_fence(block.text)
        return f"{fence}{block.lang}\n{block.text.rstrip()}\n{fence}"
    if isinstance(block, KpiRow):
        rows = [
            (strong(t.label), f"{t.value} {t.unit}".strip(), t.detail)
            for t in block.tiles
        ]
        return md_table(block.headers, rows)
    raise TypeError(f"unknown block {type(block).__name__}")


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------


def esc(text: str) -> str:
    """HTML-escape ``text``; ``://`` is entity-encoded so no URL appears."""
    return html.escape(text, quote=True).replace("://", ":&#47;&#47;")


def html_inline(content: Cell) -> str:
    """HTML for inline content."""
    out: list[str] = []
    for span in inline(content):
        text = esc(span.text)
        if span.style == "code":
            out.append(f"<code>{text}</code>")
        elif span.style == "strong":
            out.append(f"<strong>{text}</strong>")
        elif span.style == "em":
            out.append(f"<em>{text}</em>")
        else:
            out.append(text)
    return "".join(out)


def _html_table(block: Table) -> str:
    def cls(index: int) -> str:
        return ' class="num"' if index in block.numeric else ""

    head = "".join(
        f'<th{cls(i)} scope="col">{esc(h)}</th>' for i, h in enumerate(block.headers)
    )
    body_rows = []
    for row in block.rows:
        cells = "".join(f"<td{cls(i)}>{html_inline(c)}</td>" for i, c in enumerate(row))
        body_rows.append(f"<tr>{cells}</tr>")
    caption = f"<caption>{esc(block.caption)}</caption>" if block.caption else ""
    return (
        f'<div class="tbl"><table>{caption}<thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _html_block(doc: Document, block: Block) -> str:
    if isinstance(block, Heading):
        level = min(max(block.level, 2), 4)
        anchor = f' id="{esc(block.anchor)}"' if block.anchor else ""
        return f"<h{level}{anchor}>{esc(block.text)}</h{level}>"
    if isinstance(block, Paragraph):
        body = html_inline(block.content)
        if block.tone == "normal":
            return f"<p>{body}</p>"
        return f'<p class="callout {block.tone}">{body}</p>'
    if isinstance(block, Bullets):
        items = "".join(f"<li>{html_inline(item)}</li>" for item in block.items)
        return f"<ul>{items}</ul>"
    if isinstance(block, Table):
        return _html_table(block)
    if isinstance(block, Figure):
        return (
            f'<figure class="viz" id="{esc(block.chart_id)}">{block.svg}'
            f"<figcaption>{esc(doc.figure_caption(block))}</figcaption></figure>"
        )
    if isinstance(block, Code):
        lang = f' class="lang-{esc(block.lang)}"' if block.lang else ""
        return f"<pre><code{lang}>{esc(block.text.rstrip())}</code></pre>"
    if isinstance(block, KpiRow):
        tiles = []
        for tile in block.tiles:
            detail = (
                f'<div class="detail">{esc(tile.detail)}</div>' if tile.detail else ""
            )
            unit = f'<span class="unit">{esc(tile.unit)}</span>' if tile.unit else ""
            tiles.append(
                f'<div class="kpi"><div class="label">{esc(tile.label)}</div>'
                f'<div class="value">{esc(tile.value)}{unit}</div>{detail}</div>'
            )
        return f'<div class="kpis">{"".join(tiles)}</div>'
    raise TypeError(f"unknown block {type(block).__name__}")


def render_html(doc: Document, *, extra_css: str = "") -> str:
    """Render ``doc`` as one self-contained HTML page (inline CSS and SVG)."""
    toc_items = "".join(
        f'<li><a href="#{esc(b.anchor)}">{esc(b.text)}</a></li>'
        for b in doc.blocks
        if isinstance(b, Heading) and b.level == 2 and b.anchor
    )
    toc = (
        f'<nav class="toc"><div class="toc-title">{esc(doc.toc_label)}</div>'
        f"<ul>{toc_items}</ul></nav>"
        if toc_items
        else ""
    )
    body = "\n".join(_html_block(doc, block) for block in doc.blocks)
    footer = f"<footer>{esc(doc.footer)}</footer>" if doc.footer else ""
    subtitle = f'<p class="subtitle">{esc(doc.subtitle)}</p>' if doc.subtitle else ""
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="{esc(doc.lang)}">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(doc.title)}</title>\n"
        f"<style>\n{BASE_CSS}{extra_css}</style>\n</head>\n<body>\n<main>\n"
        f"<header><h1>{esc(doc.title)}</h1>{subtitle}</header>\n{toc}\n"
        f"{body}\n{footer}\n</main>\n</body>\n</html>\n"
    )


FONT_STACK = (
    'system-ui, -apple-system, "Segoe UI", "PingFang SC", "Hiragino Sans GB", '
    '"Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", '
    '"WenQuanYi Micro Hei", sans-serif'
)
MONO_STACK = (
    'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", '
    '"Noto Sans Mono CJK SC", monospace'
)

BASE_CSS = f"""\
:root {{
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --wash: rgba(11, 11, 11, 0.045); --note: #86b6ef; --note-bg: #eef4fc;
  --warn: #eda100; --warn-bg: #fdf5e2;
}}
* {{ box-sizing: border-box; }}
html {{ font-size: 15px; -webkit-text-size-adjust: 100%; }}
body {{ margin: 0; background: var(--page); color: var(--ink);
  font-family: {FONT_STACK}; line-height: 1.65; }}
main {{ max-width: 1000px; margin: 0 auto; padding: 32px 30px 56px;
  background: var(--surface); }}
header h1 {{ font-size: 1.6rem; line-height: 1.3; margin: 0 0 6px; font-weight: 650; }}
.subtitle {{ color: var(--ink2); margin: 0 0 18px; font-size: .9rem; }}
h2 {{ font-size: 1.25rem; margin: 2.2em 0 .6em; padding-bottom: .25em;
  border-bottom: 1px solid var(--grid); }}
h3 {{ font-size: 1.05rem; margin: 1.6em 0 .5em; }}
p {{ margin: .55em 0; }}
ul {{ margin: .4em 0 .8em; padding-left: 1.4em; }}
li {{ margin: .2em 0; }}
code, pre {{ font-family: {MONO_STACK}; font-size: .86em; }}
code {{ background: var(--wash); padding: 0 .3em; border-radius: 4px;
  overflow-wrap: anywhere; }}
pre {{ background: var(--wash); border: 1px solid var(--grid); border-radius: 6px;
  padding: 10px 12px; overflow-x: auto; white-space: pre-wrap;
  overflow-wrap: anywhere; line-height: 1.5; }}
pre code {{ background: none; padding: 0; }}
.tbl {{ overflow-x: auto; margin: .8em 0 1.2em; }}
table {{ border-collapse: collapse; width: 100%; font-size: .88rem; }}
th, td {{ border-bottom: 1px solid var(--grid); padding: 5px 8px; text-align: left;
  vertical-align: top; }}
th {{ color: var(--ink2); font-weight: 600; border-bottom-color: var(--axis); }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums;
  white-space: nowrap; }}
caption {{ caption-side: top; text-align: left; color: var(--ink2);
  font-size: .85rem; padding: 0 0 4px; }}
nav.toc {{ border: 1px solid var(--grid); border-radius: 8px; padding: 10px 16px;
  margin: 8px 0 20px; font-size: .9rem; }}
nav.toc .toc-title {{ font-weight: 600; color: var(--ink2); }}
nav.toc ul {{ columns: 2; margin: 4px 0 0; padding-left: 1.1em; }}
nav.toc a {{ color: var(--ink); text-decoration: none; }}
nav.toc a:hover {{ text-decoration: underline; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
  gap: 10px; margin: 1em 0 1.2em; }}
.kpi {{ border: 1px solid var(--grid); border-radius: 8px; padding: 10px 12px; }}
.kpi .label {{ color: var(--ink2); font-size: .82rem; }}
.kpi .value {{ font-size: 1.4rem; font-weight: 650; line-height: 1.3; }}
.kpi:first-child .value {{ font-size: 2.1rem; }}
.kpi .unit {{ font-size: .8rem; font-weight: 500; color: var(--ink2);
  margin-left: .3em; }}
.kpi .detail {{ color: var(--ink2); font-size: .78rem; }}
.callout {{ border-left: 3px solid var(--note); background: var(--note-bg);
  padding: 8px 12px; border-radius: 0 6px 6px 0; }}
.callout.warn {{ border-left-color: var(--warn); background: var(--warn-bg); }}
figure.viz {{ margin: 1.2em 0 1.6em; }}
figure.viz svg {{ display: block; width: 100%; max-width: 760px; height: auto; }}
figcaption {{ color: var(--ink2); font-size: .85rem; margin-top: 4px; }}
footer {{ margin-top: 3em; color: var(--muted); font-size: .8rem;
  border-top: 1px solid var(--grid); padding-top: 10px; }}
@media screen and (prefers-color-scheme: dark) {{
  :root {{ color-scheme: dark; --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec;
    --ink2: #c3c2b7; --grid: #2c2c2a; --axis: #383835;
    --wash: rgba(255, 255, 255, 0.06); --note-bg: #16253a; --warn-bg: #2e2510; }}
}}
@media print {{
  @page {{ size: A4; margin: 14mm 12mm; }}
  html {{ font-size: 10.5pt; }}
  body {{ background: #fff; }}
  main {{ max-width: none; padding: 0; background: #fff; }}
  nav.toc {{ display: none; }}
  h2, h3 {{ break-after: avoid; page-break-after: avoid; }}
  figure.viz, .kpis, tr, pre {{ break-inside: avoid; page-break-inside: avoid; }}
  .tbl {{ overflow: visible; }}
  * {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
}}
"""
