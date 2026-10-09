"""Deterministic pre-research triage for requests the assistant must never act on.

Why this exists: the data is read-only, but a request like "drop the sales table" was
being routed to the *web researcher*, which spent minutes searching DROP TABLE syntax
for five different databases before eventually refusing. The tool-level guards were
never the bottleneck - the request should be refused before any research starts.

This is a cheap first line (zero LLM calls, ~microseconds), not the security boundary:
paraphrases can slip past a regex, and when they do the SQL guard, read-only connection
and sandbox still stop the action. False positives are controlled by an exemption list
(questions *about* SQL, or about records that were already deleted) and by a test that
asserts none of the legitimate evaluation questions is refused.
"""

from __future__ import annotations

import re
from typing import Optional

# ---- English / SQL
_SQL_MUTATION = re.compile(
    r"\b(drop\s+(table|database|schema|index|view)|truncate\s+table|delete\s+from|insert\s+into|"
    r"update\s+\w+\s+set|alter\s+table|create\s+table|grant\s+all|revoke)\b", re.I)
_EN_INTENT = re.compile(
    r"\b(delete|remove|erase|wipe|clear|modify|change|set|update|insert|add|overwrite)\b"
    r".{0,40}\b(table|database|rows?|records?|orders?|data|revenue|units)\b", re.I)

# ---- Chinese
_ZH_VERB = r"(删除|删掉|清空|清除|移除|抹掉|修改|改成|改为|更改|篡改|更新|新增|添加|插入|写入|覆盖)"
_ZH_OBJECT = r"(表|数据库|数据|订单|记录|字段|销售额|销量|revenue|units|sales|products|产品线|大区)"
_ZH_MUTATION = re.compile(_ZH_VERB + r".{0,24}" + _ZH_OBJECT + r"|把.{0,40}" + _ZH_OBJECT + r".{0,24}" + _ZH_VERB)
_ZH_BENIGN_UPDATE = re.compile(r"更新(时间|日期|频率|日志)")
_ZH_NEGATED = re.compile(r"(被|已|曾|未|没有|是否|有没有)\s*" + _ZH_VERB)       # "被删除的订单有几条"
_ZH_EXPORT = re.compile(r"(导出|下载|dump|export).{0,12}(所有|全部|明细|手机号|客户|薪资|提成明细)|"
                        r"(所有|全部|明细|手机号|客户|薪资).{0,16}(导出|下载)")

# ---- prompt injection that tries to switch the rules off
_INJECTION = re.compile(
    r"(忽略|无视|忘掉|忘记|不要遵守).{0,8}(之前|以上|先前|所有|全部).{0,8}(规则|指令|提示|设定|限制)|"
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+(instructions|rules)|"
    r"disregard\s+(all\s+)?(previous|prior|your)\s+(instructions|rules)", re.I)

# questions *about* SQL / past events are fine
_EXEMPT = re.compile(r"(什么是|是什么意思|解释|区别|语法|怎么写|how\s+does|what\s+is|what\s+does|explain|difference|syntax)", re.I)

_REFUSAL_ZH = ("该请求要求{what}，而本助手对业务数据只有只读权限，已拒绝执行，数据没有任何变化。"
               "我可以帮你查询、统计或分析现有数据，需要的话请告诉我具体想了解什么。")
_REFUSAL_EN = ("This request asks to {what}, but this assistant has read-only access to business data. "
               "It was refused and nothing was changed. I can query, summarise or analyse the existing data instead.")


def classify_request(text: str) -> Optional[str]:
    """Return a short reason code if ``text`` must be refused, else None."""
    if not text or not text.strip():
        return None
    if _INJECTION.search(text):
        return "prompt_injection"
    exempt = bool(_EXEMPT.search(text))
    if _SQL_MUTATION.search(text) and not exempt:
        return "data_mutation"
    if _EN_INTENT.search(text) and not exempt and not re.search(r"\b(deleted|removed|cleared)\b", text, re.I):
        return "data_mutation"
    zh = _ZH_BENIGN_UPDATE.sub("", text)
    if _ZH_MUTATION.search(zh) and not _ZH_NEGATED.search(zh) and not exempt:
        return "data_mutation"
    if _ZH_EXPORT.search(text):
        return "bulk_export"
    return None


def refusal_message(reason: str, text: str) -> str:
    what_zh = {"data_mutation": "修改、删除或新增数据", "bulk_export": "批量导出机密明细数据",
               "prompt_injection": "绕过既有的安全规则"}[reason]
    what_en = {"data_mutation": "modify, delete or insert data", "bulk_export": "bulk-export confidential detail",
               "prompt_injection": "bypass the existing safety rules"}[reason]
    if re.search(r"[一-鿿]", text):
        return _REFUSAL_ZH.format(what=what_zh)
    return _REFUSAL_EN.format(what=what_en)
