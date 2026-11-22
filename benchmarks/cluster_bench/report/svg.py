# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiny dependency-free SVG charts for the cluster_bench report.

Visual rules: one shared y-axis; solid hairline grid; 2px lines with ringed
markers and one marker shape per series (a second channel for grayscale
print and colour-vision deficiency); bars at most 24px thick with a 4px
rounded data end and a square baseline end; a legend for two or more
series and only selective direct labels. Every mark carries a ``<title>``
so browsers show a native tooltip without any script, and every chart in
the report has a table with the same numbers next to it.

Colours are the validated categorical order of the report palette, assigned
by series position, never cycled: at most ``MAX_SERIES`` series are drawn.
Each element carries both a light-theme presentation attribute (so a
standalone ``.svg`` renders correctly) and a class that ``theme_css`` can
restyle for a dark screen.

Charts are returned without ``xmlns`` so they can be inlined in HTML5 with
no URL in the page; ``standalone`` adds the namespace for a ``.svg`` file.
Output is a pure function of the inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from benchmarks.cluster_bench.report.doc import esc

SVG_NS = "http://www.w3.org/2000/svg"
FONT = (
    "system-ui, -apple-system, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', "
    "'Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans SC', sans-serif"
)

# (light, dark) steps of the categorical slots, in their validated order.
SERIES = (
    ("#2a78d6", "#3987e5"),
    ("#eb6834", "#d95926"),
    ("#1baf7a", "#199e70"),
    ("#eda100", "#c98500"),
    ("#e87ba4", "#d55181"),
    ("#008300", "#008300"),
    ("#6250d6", "#9085e9"),
    ("#e34948", "#e66767"),
)
MAX_SERIES = len(SERIES)
SURFACE = ("#fcfcfb", "#1a1a19")
INK = ("#0b0b0b", "#f0efec")
INK2 = ("#52514e", "#c3c2b7")
MUTED = ("#898781", "#898781")
GRID = ("#e1e0d9", "#2c2c2a")
AXIS = ("#c3c2b7", "#383835")

TICK_FS = 11
LABEL_FS = 12
BAR_MAX = 24.0
RADIUS = 4.0


@dataclass(frozen=True)
class Point:
    """One data point; ``tip`` is the hover text."""

    x: float
    y: float
    tip: str = ""
    label: str = ""  # direct label next to the marker; use sparingly


@dataclass(frozen=True)
class Series:
    """A named sequence of points (one line)."""

    name: str
    points: tuple[Point, ...]


@dataclass(frozen=True)
class RefLine:
    """A dashed reference line (threshold) with an optional label."""

    value: float
    label: str = ""


@dataclass(frozen=True)
class Bar:
    """One horizontal bar; ``muted`` draws it in the neutral gray."""

    label: str
    value: float | None
    value_label: str = ""
    tip: str = ""
    muted: bool = False


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def _f(value: float) -> str:
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def text_width(text: str, size: float) -> float:
    """Rough rendered width of ``text`` (CJK glyphs are one em wide)."""
    total = 0.0
    for ch in text:
        if ord(ch) >= 0x2E80:
            total += size
        elif ch in "il.,:;|!'·()[] ":
            total += size * 0.32
        elif ch.isupper() or ch in "mwMW%@":
            total += size * 0.74
        else:
            total += size * 0.56
    return total


def nice_ticks(lo: float, hi: float, target: int = 5) -> tuple[float, ...]:
    """Round 1-2-2.5-5 ticks covering ``[lo, hi]``."""
    if not hi > lo:
        hi = lo + 1.0
    raw = (hi - lo) / target
    mag = 10.0 ** math.floor(math.log10(raw))
    step = 10.0 * mag
    for mantissa in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= mantissa * mag * (1 + 1e-9):
            step = mantissa * mag
            break
    start = math.floor(lo / step + 1e-9)
    end = math.ceil(hi / step - 1e-9)
    return tuple(round(k * step, 10) for k in range(start, end + 1))


def tick_label(value: float, ticks: Sequence[float]) -> str:
    """Tick text with as many decimals as the tick step needs."""
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
    exponent = math.floor(math.log10(step)) if step > 0 else 0
    mantissa = step / (10.0**exponent) if step > 0 else 1.0
    decimals = max(0, -exponent + (1 if abs(mantissa - 2.5) < 1e-6 else 0))
    return f"{value:,.{decimals}f}"


def _text(
    x: float,
    y: float,
    content: str,
    *,
    size: int = TICK_FS,
    anchor: str = "start",
    tone: str = "tx",
    weight: int | None = None,
    rotate: bool = False,
) -> str:
    fill = INK[0] if tone == "tx1" else (MUTED[0] if tone == "mut" else INK2[0])
    extra = f' font-weight="{weight}"' if weight else ""
    if rotate:
        extra += f' transform="rotate(-90 {_f(x)} {_f(y)})"'
    return (
        f'<text class="{tone}" x="{_f(x)}" y="{_f(y)}" font-size="{size}" '
        f'text-anchor="{anchor}" fill="{fill}"{extra}>{esc(content)}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, cls: str) -> str:
    colors = {"grid": GRID[0], "axis": AXIS[0], "ref": MUTED[0]}
    dash = ' stroke-dasharray="4 3"' if cls == "ref" else ""
    return (
        f'<line class="{cls}" x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" '
        f'y2="{_f(y2)}" stroke="{colors[cls]}" stroke-width="1"{dash}/>'
    )


def _marker(shape: int, x: float, y: float, slot: int) -> str:
    color = SERIES[slot][0]
    attrs = (
        f'class="mk s{slot + 1} ring" fill="{color}" stroke="{SURFACE[0]}" '
        'stroke-width="2"'
    )
    r = 4.5
    if shape == 0:
        return f'<circle {attrs} cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}"/>'
    if shape == 1:
        s = r * 0.9
        return (
            f'<rect {attrs} x="{_f(x - s)}" y="{_f(y - s)}" '
            f'width="{_f(2 * s)}" height="{_f(2 * s)}"/>'
        )
    if shape == 2:
        d = r * 1.3
        path = f"M{_f(x)},{_f(y - d)}L{_f(x + d)},{_f(y)}L{_f(x)},{_f(y + d)}"
        return f'<path {attrs} d="{path}L{_f(x - d)},{_f(y)}Z"/>'
    sign = 1 if shape == 3 else -1
    top, base = y - sign * r * 1.3, y + sign * r * 0.9
    path = f"M{_f(x)},{_f(top)}L{_f(x + r * 1.2)},{_f(base)}"
    return f'<path {attrs} d="{path}L{_f(x - r * 1.2)},{_f(base)}Z"/>'


def _hit(x: float, y: float, tip: str, body: str) -> str:
    title = f"<title>{esc(tip)}</title>" if tip else ""
    return (
        f'<g class="pt">{title}<circle cx="{_f(x)}" cy="{_f(y)}" r="12" '
        f'fill="#000" fill-opacity="0"/>{body}</g>'
    )


def _open(chart_id: str, title: str, width: float, height: float) -> list[str]:
    return [
        f'<svg class="chart" width="{_f(width)}" height="{_f(height)}" '
        f'viewBox="0 0 {_f(width)} {_f(height)}" role="img" '
        f'aria-labelledby="{esc(chart_id)}-title" font-family="{FONT}">',
        f'<title id="{esc(chart_id)}-title">{esc(title)}</title>',
        f'<rect class="bg" x="0" y="0" width="{_f(width)}" height="{_f(height)}" '
        f'fill="{SURFACE[0]}"/>',
    ]


def _legend(
    names: Sequence[str], x0: float, x1: float, y: float, kind: str
) -> tuple[str, int]:
    """Lay out legend items left to right, wrapping rows; returns (svg, rows)."""
    if len(names) < 2:
        return "", 0
    parts: list[str] = []
    cx, row = x0, 0
    for slot, name in enumerate(names):
        item_w = 26 + text_width(name, LABEL_FS) + 18
        if cx + item_w > x1 and cx > x0:
            cx, row = x0, row + 1
        cy = y + row * 20
        color = SERIES[slot][0]
        if kind == "line":
            parts.append(
                f'<line class="ln s{slot + 1}" x1="{_f(cx)}" y1="{_f(cy)}" '
                f'x2="{_f(cx + 20)}" y2="{_f(cy)}" stroke="{color}" '
                'stroke-width="2" stroke-linecap="round"/>'
            )
            parts.append(_marker(slot % 5, cx + 10, cy, slot))
        else:
            parts.append(
                f'<rect class="mk s{slot + 1}" x="{_f(cx + 4)}" y="{_f(cy - 6)}" '
                f'width="12" height="12" rx="3" fill="{color}"/>'
            )
        parts.append(_text(cx + 26, cy + 4, name, size=LABEL_FS, tone="tx1"))
        cx += item_w
    return "".join(parts), row + 1


def _empty(chart_id: str, title: str, message: str, width: float) -> str:
    height = 120.0
    parts = _open(chart_id, title, width, height)
    parts.append(_line(40, height - 30, width - 20, height - 30, "axis"))
    parts.append(_text(width / 2, height / 2, message, size=LABEL_FS, anchor="middle"))
    parts.append("</svg>")
    return "".join(parts)


def _column_path(x: float, w: float, top: float, base: float) -> str:
    r = min(RADIUS, w / 2, max(base - top, 0.0))
    return (
        f"M{_f(x)},{_f(base)}V{_f(top + r)}Q{_f(x)},{_f(top)} {_f(x + r)},{_f(top)}"
        f"H{_f(x + w - r)}Q{_f(x + w)},{_f(top)} {_f(x + w)},{_f(top + r)}"
        f"V{_f(base)}Z"
    )


def _bar_path(x0: float, x1: float, y: float, h: float) -> str:
    r = min(RADIUS, h / 2, max(x1 - x0, 0.0))
    return (
        f"M{_f(x0)},{_f(y)}H{_f(x1 - r)}Q{_f(x1)},{_f(y)} {_f(x1)},{_f(y + r)}"
        f"V{_f(y + h - r)}Q{_f(x1)},{_f(y + h)} {_f(x1 - r)},{_f(y + h)}"
        f"H{_f(x0)}Z"
    )


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def line_chart(
    chart_id: str,
    series: Sequence[Series],
    *,
    title: str,
    x_label: str,
    y_label: str,
    log2_x: bool = False,
    x_refs: Sequence[RefLine] = (),
    empty_text: str = "",
    width: float = 720.0,
    height: float = 380.0,
) -> str:
    """Multi-series line chart; ``log2_x`` puts x on a log2 axis (concurrency)."""
    drawn = [s for s in series if s.points][:MAX_SERIES]
    if log2_x:
        drawn = [Series(s.name, tuple(p for p in s.points if p.x > 0)) for s in drawn]
        drawn = [s for s in drawn if s.points]
    if not drawn:
        return _empty(chart_id, title, empty_text, width)

    xs = [p.x for s in drawn for p in s.points]
    ys = [p.y for s in drawn for p in s.points]
    y_ticks = nice_ticks(0.0, max(max(ys), 0.0) * 1.08 or 1.0)
    y_max = y_ticks[-1]

    if log2_x:
        lo, hi = math.log2(min(xs)), math.log2(max(xs))
        pad = 0.5 if hi == lo else 0.06 * (hi - lo) + 0.15
        lo, hi = lo - pad, hi + pad
        ks = list(range(math.ceil(lo), math.floor(hi) + 1))
        if len(ks) > 9:
            ks = ks[:: math.ceil(len(ks) / 9)]
        x_ticks = [(2.0**k, str(2**k) if k >= 0 else f"{2.0**k:g}") for k in ks]

        def tx(v: float) -> float:
            return math.log2(v)

    else:
        limit = max(xs + [r.value for r in x_refs]) * 1.05
        ticks = nice_ticks(0.0, limit or 1.0)
        lo, hi = 0.0, ticks[-1]
        x_ticks = [(t, tick_label(t, ticks)) for t in ticks]

        def tx(v: float) -> float:
            return v

    y_label_w = max(text_width(tick_label(t, y_ticks), TICK_FS) for t in y_ticks)
    x0 = 30 + y_label_w + 8
    x1 = width - 24
    legend_svg, rows = _legend([s.name for s in drawn], x0, x1, 14, "line")
    labelled_refs = any(r.label for r in x_refs)
    y0 = 14 + rows * 20 + (10 if rows else 6) + (16 if labelled_refs else 0)
    y1 = height - 50

    def sx(v: float) -> float:
        return x0 + (tx(v) - lo) / (hi - lo) * (x1 - x0)

    def sy(v: float) -> float:
        return y1 - v / y_max * (y1 - y0)

    parts = _open(chart_id, title, width, height)
    for t in y_ticks:
        cls = "axis" if t == 0 else "grid"
        parts.append(_line(x0, sy(t), x1, sy(t), cls))
        parts.append(
            _text(x0 - 6, sy(t) + 4, tick_label(t, y_ticks), anchor="end", tone="tx")
        )
    for value, label in x_ticks:
        px = sx(value)
        parts.append(_line(px, y1, px, y1 + 4, "axis"))
        parts.append(_text(px, y1 + 17, label, anchor="middle"))
    for ref in x_refs:
        if (ref.value > 0 or not log2_x) and lo <= tx(ref.value) <= hi:
            px = sx(ref.value)
            parts.append(_line(px, y0, px, y1, "ref"))
            if ref.label:
                parts.append(_text(px, y0 - 5, ref.label, size=10, anchor="middle"))
    parts.append(
        _text((x0 + x1) / 2, height - 12, x_label, size=LABEL_FS, anchor="middle")
    )
    parts.append(
        _text(14, (y0 + y1) / 2, y_label, size=LABEL_FS, anchor="middle", rotate=True)
    )
    parts.append(legend_svg)

    labels: list[str] = []
    for slot, s in enumerate(drawn):
        color = SERIES[slot][0]
        coords = [(sx(p.x), sy(p.y)) for p in s.points]
        if len(coords) > 1:
            d = "M" + "L".join(f"{_f(x)},{_f(y)}" for x, y in coords)
            parts.append(
                f'<path class="ln s{slot + 1}" d="{d}" fill="none" stroke="{color}" '
                'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        for p, (px, py) in zip(s.points, coords):
            parts.append(_hit(px, py, p.tip, _marker(slot % 5, px, py, slot)))
            if p.label:
                right = px + 8 + text_width(p.label, TICK_FS) > x1
                labels.append(
                    _text(
                        px - 8 if right else px + 8,
                        py - 9,
                        p.label,
                        anchor="end" if right else "start",
                        tone="tx1",
                        weight=600,
                    )
                )
    parts.extend(labels)
    parts.append("</svg>")
    return "".join(parts)


def hbar_chart(
    chart_id: str,
    bars: Sequence[Bar],
    *,
    title: str,
    x_label: str,
    ref: RefLine | None = None,
    empty_text: str = "",
    width: float = 720.0,
) -> str:
    """Horizontal bars on one baseline, value labels at the bar tips."""
    if not bars:
        return _empty(chart_id, title, empty_text, width)
    row_h, thick = 34.0, 18.0
    label_w = max(text_width(b.label, LABEL_FS) for b in bars)
    value_w = max(text_width(b.value_label, TICK_FS) for b in bars)
    x0 = 12 + label_w + 12
    x1 = width - 16 - value_w - 8
    top = 12.0
    y_end = top + row_h * len(bars)
    height = y_end + 46
    values = [b.value for b in bars if b.value is not None and b.value > 0]
    if ref is not None:
        values.append(ref.value)
    ticks = nice_ticks(0.0, max(values) if values else 1.0)
    x_max = ticks[-1]

    def sx(v: float) -> float:
        return x0 + max(v, 0.0) / x_max * (x1 - x0)

    parts = _open(chart_id, title, width, height)
    for t in ticks:
        parts.append(_line(sx(t), top, sx(t), y_end, "axis" if t == 0 else "grid"))
        parts.append(_text(sx(t), y_end + 16, tick_label(t, ticks), anchor="middle"))
    if ref is not None:
        parts.append(_line(sx(ref.value), top, sx(ref.value), y_end, "ref"))
    for index, bar in enumerate(bars):
        cy = top + row_h * (index + 0.5)
        parts.append(
            _text(x0 - 8, cy + 4, bar.label, size=LABEL_FS, anchor="end", tone="tx1")
        )
        if bar.value is None or bar.value <= 0:
            body = _text(x0 + 6, cy + 4, bar.value_label, tone="mut")
            parts.append(_hit(x0 + 6, cy, bar.tip, body))
            continue
        cls, color = ("mut", MUTED[0]) if bar.muted else ("s1", SERIES[0][0])
        path = _bar_path(x0, sx(bar.value), cy - thick / 2, thick)
        body = f'<path class="mk {cls}" d="{path}" fill="{color}"/>'
        title_el = f"<title>{esc(bar.tip)}</title>" if bar.tip else ""
        parts.append(f'<g class="bar">{title_el}{body}</g>')
        parts.append(_text(sx(bar.value) + 6, cy + 4, bar.value_label))
    parts.append(
        _text((x0 + x1) / 2, height - 10, x_label, size=LABEL_FS, anchor="middle")
    )
    parts.append("</svg>")
    return "".join(parts)


def grouped_bar_chart(
    chart_id: str,
    groups: Sequence[str],
    series_names: Sequence[str],
    values: Sequence[Sequence[float | None]],
    *,
    title: str,
    y_label: str,
    tips: Sequence[Sequence[str]] | None = None,
    value_labels: Sequence[Sequence[str]] | None = None,
    ref: RefLine | None = None,
    empty_text: str = "",
    width: float = 720.0,
    height: float = 340.0,
) -> str:
    """Vertical grouped columns: ``values[group][series]``."""
    names = list(series_names)[:MAX_SERIES]
    flat = [v for row in values for v in row[: len(names)] if v is not None and v > 0]
    if not groups or not names or not flat:
        return _empty(chart_id, title, empty_text, width)
    if ref is not None:
        flat.append(ref.value)
    y_ticks = nice_ticks(0.0, max(flat) * 1.1)
    y_max = y_ticks[-1]
    y_label_w = max(text_width(tick_label(t, y_ticks), TICK_FS) for t in y_ticks)
    x0 = 30 + y_label_w + 8
    x1 = width - 16
    legend_svg, rows = _legend(names, x0, x1, 14, "bar")
    y0 = 14 + rows * 20 + (12 if rows else 8)
    y1 = height - 56

    def sy(v: float) -> float:
        return y1 - max(v, 0.0) / y_max * (y1 - y0)

    parts = _open(chart_id, title, width, height)
    for t in y_ticks:
        parts.append(_line(x0, sy(t), x1, sy(t), "axis" if t == 0 else "grid"))
        parts.append(_text(x0 - 6, sy(t) + 4, tick_label(t, y_ticks), anchor="end"))
    if ref is not None:
        parts.append(_line(x0, sy(ref.value), x1, sy(ref.value), "ref"))
        if ref.label:
            parts.append(_text(x1, sy(ref.value) - 4, ref.label, size=10, anchor="end"))
    parts.append(
        _text(14, (y0 + y1) / 2, y_label, size=LABEL_FS, anchor="middle", rotate=True)
    )
    parts.append(legend_svg)
    group_w = (x1 - x0) / len(groups)
    gap = 2.0
    bar_w = min(BAR_MAX, (group_w * 0.72 - gap * (len(names) - 1)) / len(names))
    cluster = bar_w * len(names) + gap * (len(names) - 1)
    for g_index, group in enumerate(groups):
        center = x0 + group_w * (g_index + 0.5)
        for line_no, text in enumerate(group.split("\n")[:2]):
            parts.append(_text(center, y1 + 17 + 14 * line_no, text, anchor="middle"))
        row = values[g_index] if g_index < len(values) else ()
        for s_index in range(len(names)):
            value = row[s_index] if s_index < len(row) else None
            bx = center - cluster / 2 + s_index * (bar_w + gap)
            tip = tips[g_index][s_index] if tips else ""
            if value is None:
                body = _text(bx + bar_w / 2, y1 - 4, "—", anchor="middle", tone="mut")
                parts.append(_hit(bx + bar_w / 2, y1 - 8, tip, body))
                continue
            label = value_labels[g_index][s_index] if value_labels else ""
            if value <= 0:
                if label:
                    parts.append(
                        _text(bx + bar_w / 2, y1 - 5, label, size=10, anchor="middle")
                    )
                continue
            color = SERIES[s_index][0]
            path = _column_path(bx, bar_w, sy(value), y1)
            title_el = f"<title>{esc(tip)}</title>" if tip else ""
            parts.append(
                f'<g class="bar">{title_el}<path class="mk s{s_index + 1}" '
                f'd="{path}" fill="{color}"/></g>'
            )
            if label:
                parts.append(
                    _text(
                        bx + bar_w / 2, sy(value) - 5, label, size=10, anchor="middle"
                    )
                )
    parts.append("</svg>")
    return "".join(parts)


def standalone(svg: str) -> str:
    """Turn an inline chart into a standalone ``.svg`` document."""
    if not svg.startswith("<svg "):
        raise ValueError("not an svg element")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="{SVG_NS}" {svg[len("<svg "):]}\n'
    )


def theme_css() -> str:
    """CSS that restyles inline charts: hover lift and the dark-screen theme."""
    dark = [
        f".viz .bg{{fill:{SURFACE[1]}}}",
        f".viz .ring{{stroke:{SURFACE[1]}}}",
        f".viz .grid{{stroke:{GRID[1]}}}",
        f".viz .axis{{stroke:{AXIS[1]}}}",
        f".viz .ref{{stroke:{MUTED[1]}}}",
        f".viz .tx{{fill:{INK2[1]}}}",
        f".viz .tx1{{fill:{INK[1]}}}",
        f".viz .mut,.viz .mk.mut{{fill:{MUTED[1]}}}",
    ]
    for slot, (_, color) in enumerate(SERIES, start=1):
        dark.append(f".viz .ln.s{slot}{{stroke:{color}}}")
        dark.append(f".viz .mk.s{slot}{{fill:{color}}}")
    rules = "\n".join(f"  {rule}" for rule in dark)
    return (
        ".viz .pt:hover .mk, .viz .bar:hover .mk { opacity: .78; }\n"
        f"@media screen and (prefers-color-scheme: dark) {{\n{rules}\n}}\n"
    )
