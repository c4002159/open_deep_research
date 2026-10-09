"""Verify that the documents a report cites really exist and were actually retrieved.

Two independent failure modes are checked:
  unknown     - the cited file is not in the knowledge base at all (fabricated source)
  ungrounded  - it exists, but nothing from it appeared in this run's research notes, so the
                model is citing from memory/imagination rather than from what it retrieved
The report is never silently rewritten; a clearly marked note is appended instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_CITED = re.compile(r"(?<![\w/.])([A-Za-z0-9][\w\-]*\.md)\b")


@dataclass
class CitationReport:
    cited: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)        # not in the knowledge base
    ungrounded: list[str] = field(default_factory=list)     # in the KB, but not retrieved in this run

    @property
    def ok(self) -> bool:
        return not self.unknown and not self.ungrounded

    @property
    def validity(self) -> float:
        return 1.0 if not self.cited else 1 - len(set(self.unknown) | set(self.ungrounded)) / len(self.cited)


def cited_documents(text: str) -> list[str]:
    seen: list[str] = []
    for m in _CITED.finditer(text or ""):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def verify_citations(report: str, findings: str, known_sources: Optional[set[str]]) -> CitationReport:
    """``known_sources=None`` means the knowledge base could not be read: only groundedness is checked."""
    cited = cited_documents(report)
    low_findings = (findings or "").lower()
    res = CitationReport(cited=cited)
    for doc in cited:
        if known_sources is not None and doc not in known_sources:
            res.unknown.append(doc)
        elif doc.lower() not in low_findings:
            res.ungrounded.append(doc)
    return res


def footer(res: CitationReport, chinese: bool = True) -> str:
    if res.ok:
        return ""
    if chinese:
        parts = []
        if res.unknown:
            parts.append("知识库中不存在：" + "、".join(res.unknown))
        if res.ungrounded:
            parts.append("本次检索结果中未出现：" + "、".join(res.ungrounded))
        return "\n\n> ⚠️ **引用核验未通过** —— " + "；".join(parts) + "。请勿将这些引用当作依据。"
    parts = []
    if res.unknown:
        parts.append("not in the knowledge base: " + ", ".join(res.unknown))
    if res.ungrounded:
        parts.append("not retrieved in this run: " + ", ".join(res.ungrounded))
    return "\n\n> ⚠️ **Citation check failed** — " + "; ".join(parts) + ". Do not rely on these references."
