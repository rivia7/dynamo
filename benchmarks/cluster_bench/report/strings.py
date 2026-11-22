# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loader for the report's user-facing strings (``i18n/<locale>.yaml``).

Keys are dotted paths (``sections.summary``). Templates use ``str.format``
placeholders. A missing template key raises ``KeyError`` (fail fast); enum
labels (``label``) fall back to the raw value so a new status added to the
schema degrades to its identifier instead of crashing the report.
"""

from __future__ import annotations

import re
import string
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from benchmarks.cluster_bench.report.doc import Inline, Span, inline

I18N_DIR = Path(__file__).resolve().parent / "i18n"
DEFAULT_LOCALE = "zh_CN"
_LOCALE_RE = re.compile(r"^[A-Za-z]{2,3}(_[A-Za-z]{2,4})?$")


def _text_of(value: object) -> str:
    if isinstance(value, Span):
        return value.text
    if isinstance(value, tuple) and all(isinstance(v, Span) for v in value):
        return "".join(v.text for v in value)
    return str(value)


class Strings:
    """Dotted-key access to one locale's string table."""

    def __init__(self, table: Mapping[str, Any], locale: str):
        self._table = table
        self.locale = locale

    @classmethod
    def load(cls, locale: str = DEFAULT_LOCALE) -> Strings:
        """Load (and cache) ``i18n/<locale>.yaml``."""
        return _load(locale)

    def raw(self, key: str) -> Any:
        node: Any = self._table
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise KeyError(f"missing i18n key {key!r} in locale {self.locale}")
            node = node[part]
        return node

    def _template(self, key: str) -> str:
        template = self.raw(key)
        if not isinstance(template, str):
            raise TypeError(f"i18n key {key!r} is not a string template")
        return template

    def t(self, key: str, /, **values: object) -> str:
        """Plain text: the template at ``key`` formatted with ``values``."""
        return self._template(key).format(
            **{name: _text_of(value) for name, value in values.items()}
        )

    def r(self, key: str, /, **values: object) -> Inline:
        """Rich text: ``Span``/inline values keep their style (e.g. bold numbers)."""
        spans: list[Span] = []
        for literal, field, spec, _ in string.Formatter().parse(self._template(key)):
            if literal:
                spans.append(Span(literal))
            if field is None:
                continue
            if field not in values:
                raise KeyError(f"i18n template {key!r} needs value {field!r}")
            value = values[field]
            if isinstance(value, (Span, tuple)):
                spans.extend(inline(value))
            else:
                spans.append(Span(format(value, spec) if spec else str(value)))
        return tuple(spans)

    def for_stage(self, stage: str) -> Strings:
        """This table with the ``stages`` overlay of ``stage`` applied.

        ``stages.replace.<stage>``: [old, new] text replacements over every
        template; ``stages.overrides.<stage>``: dotted key -> template.
        """
        if stage == "e2e":
            return self
        return _for_stage(self.locale, stage)

    def label(self, group: str, value: object) -> str:
        """Label of an enum value (``label("status", "pruned")``)."""
        mapping = self.raw(group)
        if not isinstance(mapping, Mapping):
            raise TypeError(f"i18n key {group!r} is not a label table")
        text = mapping.get(str(value))
        return text if isinstance(text, str) else str(value)


def _replaced(node: Any, pairs: list[tuple[str, str]]) -> Any:
    if isinstance(node, str):
        for old, new in pairs:
            node = node.replace(old, new)
        return node
    if isinstance(node, Mapping):
        return {k: _replaced(v, pairs) for k, v in node.items()}
    if isinstance(node, list):
        return [_replaced(v, pairs) for v in node]
    return node


@lru_cache(maxsize=8)
def _for_stage(locale: str, stage: str) -> Strings:
    base = _load(locale)
    overlay = base.raw("stages")
    pairs = [(str(a), str(b)) for a, b in (overlay.get("replace") or {}).get(stage, [])]
    table = _replaced(base._table, pairs)
    for dotted, template in ((overlay.get("overrides") or {}).get(stage) or {}).items():
        node = table
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node[part]
        if leaf not in node:
            raise KeyError(f"stage override {dotted!r} names no template")
        node[leaf] = template
    return Strings(table, locale)


@lru_cache(maxsize=4)
def _load(locale: str) -> Strings:
    if not _LOCALE_RE.match(locale):
        raise ValueError(f"invalid locale {locale!r}")
    path = I18N_DIR / f"{locale}.yaml"
    with open(path, encoding="utf-8") as fh:
        table = yaml.safe_load(fh)
    if not isinstance(table, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return Strings(table, locale)
