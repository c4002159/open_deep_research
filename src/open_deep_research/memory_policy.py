"""What the agent is allowed to remember, and what it is allowed to believe it remembered.

Long-term memory is free text written by a model and reused by later runs, so one wrong entry
quietly becomes "truth". The policy keeps only knowledge that stays true and that nothing else
can supply:

  KEEP    terminology      "'big order' means units > 100"
          user correction  "the user said the region column uses 'East', not 'E'"
          document location "refund approval limits are in refundPolicy.md section 2"

  REJECT  schema           the database already tells us; it is injected into the prompt, verified
          data values      totals, counts, averages, ranges... they change; query them
          data properties  "no NULLs", "categorical values are...", "each date appears once"
          speculation      "probably", "proxy", "assume" ...
          one-off events   "the assistant provided/recommended a query..."

The same rules run when saving (the model gets a reason and can rephrase) and when reading (so
entries stored before the policy existed stop influencing answers, without deleting anything).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_I = re.IGNORECASE

# --- things that must never be persisted
_TOOLING = re.compile(r"\b(tavily_search|sql_executor|python_executor|search_memory|add_memory|think_tool|search_knowledge_base|"
                      r"api[_ ]?key|ChatDeepSeek|validation error)\b", re.I)
_ONE_OFF = re.compile(r"\b(assistant|the user|user)\b.{0,20}\b(provided|recommended|suggested|asked|requested|wrote|gave)\b|用户(问|要求|询问)", _I)
_SPECULATION = re.compile(
    r"\b(probably|likely|possibly|might|may be|seems?|appears?|apparently|presumably|perhaps|guess|assum(e|es|ing|ed)|proxy|suppose)\b"
    r"|可能|似乎|大概|大约是|推测|假设|猜测|代理指标|应该是|估计", _I)
_DATA_VALUE = re.compile(
    r"[$¥￥]\s?\d|\d{1,3}(,\d{3})+|\d+\.\d+|\d+\s?%"
    r"|\b\d+\s*(rows?|records?|orders?|entries|lines)\b|\d+\s*(行|条记录|条订单|个订单|笔订单|条数据)"
    r"|\b(total|sum|average|mean|median|std|standard deviation|variance|count of)\b[^.\n]{0,40}\d"
    r"|(总额|总计|合计|平均|中位数|标准差|方差)[^。\n]{0,20}\d", _I)
_DATA_PROPERTY = re.compile(
    r"\bnulls?\b|\bnon-?null\b|空值|\bdistinct\b|\bduplicates?\b|重复|\borphan|\branging\b|\brange of\b|\bbetween\b.{0,25}\band\b"
    r"|\b(every|each)\s+(row|record|date|order|entry)\b|categorical|\bno\b.{0,20}\bvariants?\b|没有.{0,10}变体|取值范围|枚举|\bvalues? (are|include)\b|连续|\bnon-?negative\b|非负", _I)

# --- what a keepable fact looks like
_TERMINOLOGY = re.compile(
    r"\b(means?|refers? to|stands? for|defined as|is defined|aka|also called|alias(es)?|synonym|known as|maps? to|corresponds? to|serves? as|used as|equivalent to)\b"
    r"|指的是|是指|定义为|称为|对应|映射|别名|术语|口径|同义", _I)
_CORRECTION = re.compile(r"\b(user (said|corrected|clarified|prefers?|wants)|correction|actually)\b|用户(纠正|说明|指出|偏好|要求)|纠正", _I)
_DOC_LOCATION = re.compile(r"\.md\b|knowledge base|document|policy|specification|知识库|文档|制度|规范|章节|条款", _I)
_SCHEMA_LIKE = re.compile(
    r"\b(columns?|tables?|primary key|foreign key|join key|schema|rows?|data type|datatype|sqlite_master|information_schema)\b"
    r"|字段|列名|表结构|主键|外键|数据类型|数据库中.{0,6}表", _I)

# "<schema word> + contains/only/lacks/no..." is a claim about the database's structure, whatever else the
# sentence says. Checked BEFORE the fact types, otherwise a structure claim that happens to contain a mapping
# word ("... columns id (de-facto order id), date ...") would be mistaken for terminology.
_STRUCTURAL = re.compile(r"\b(contains?|includes?|has|have|lacks?|only|exactly|without|no|not)\b|没有|包含|仅|只有|缺少|不含", re.I)

_IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`|\b([A-Za-z]+_[A-Za-z0-9_]+)\b")


@dataclass
class Verdict:
    ok: bool
    category: str = ""
    reasons: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        return "; ".join(self.reasons)


GUIDANCE = (
    "Only save durable knowledge that cannot be re-derived: terminology/definitions, a correction the user gave, "
    "or where a topic is documented. Do NOT save the database schema (it is provided to you verified), numbers or "
    "query results (query again), data properties, guesses, or what you just did.")


def unknown_identifiers(fact: str, known: Optional[set[str]]) -> list[str]:
    if not known:
        return []
    found = {(m.group(1) or m.group(2)).lower() for m in _IDENT.finditer(fact)}
    return sorted(i for i in found if i not in known and not i.endswith(("_key", "_id_")) and i not in _TOOL_NAMES)


_TOOL_NAMES = {"sql_executor", "python_executor", "search_memory", "add_memory", "think_tool", "search_knowledge_base",
               "tavily_search", "mem0_user_id"}


def check_fact(fact: str, schema_identifiers: Optional[set[str]] = None) -> Verdict:
    """Decide whether ``fact`` may be stored / may be believed. Deterministic, no model call."""
    text = (fact or "").strip()
    if len(text) < 12:
        return Verdict(False, reasons=["too short to be a self-contained fact"])
    if len(text) > 600:
        return Verdict(False, reasons=["too long - save one short fact at a time"])

    reasons: list[str] = []
    if _TOOLING.search(text):
        reasons.append("is about the agent's own tools/configuration, which is operational state, not domain knowledge")
    if _ONE_OFF.search(text):
        reasons.append("describes what happened in one conversation, not a lasting fact")
    if _SPECULATION.search(text):
        reasons.append("contains speculation (a guess is not a fact)")
    if _DATA_VALUE.search(text):
        reasons.append("contains data values (totals, counts, averages...) that change - query them instead")
    if _DATA_PROPERTY.search(text):
        reasons.append("describes what the current data looks like (nulls, ranges, categories...) - it can change")
    if reasons:
        return Verdict(False, reasons=reasons)

    if _SCHEMA_LIKE.search(text) and _STRUCTURAL.search(text):
        return Verdict(False, "schema", ["it is a database-structure claim; the verified schema is provided automatically "
                                         "and is authoritative, so structure is never taken from memory"])

    if _TERMINOLOGY.search(text):
        category = "terminology"
    elif _CORRECTION.search(text):
        category = "correction"
    elif _DOC_LOCATION.search(text) and not _SCHEMA_LIKE.search(text):
        category = "document_location"
    elif _SCHEMA_LIKE.search(text):
        return Verdict(False, "schema", ["it is a database-structure claim; the verified schema is provided automatically "
                                         "and is authoritative, so structure is never taken from memory"])
    else:
        return Verdict(False, reasons=["not a recognised durable fact (terminology, user correction or document location)"])

    bad = unknown_identifiers(text, schema_identifiers)
    if bad:
        return Verdict(False, category, [f"refers to names that do not exist in the live database: {', '.join(bad)}"])
    return Verdict(True, category)
