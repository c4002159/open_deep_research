import asyncio
from types import SimpleNamespace

import pytest

from open_deep_research import deep_researcher as dr
from open_deep_research.citation_check import cited_documents, footer, verify_citations

KNOWN = {"refundPolicy.md", "salesCommissionRules.md", "dataPermission.md"}
FINDINGS = "[来源: refundPolicy.md]\n## 2. 退款审批权限\n低于 5,000 元由区域销售经理审批。"


def test_extracts_each_document_once_in_order():
    text = "依据 refundPolicy.md（第2节）和 [来源: salesCommissionRules.md]；再次引用 refundPolicy.md。"
    assert cited_documents(text) == ["refundPolicy.md", "salesCommissionRules.md"]


def test_ignores_paths_urls_and_prose():
    assert cited_documents("see https://example.com/readme.md and ./docs/x.md and no docs here") == []


def test_all_citations_valid():
    r = verify_citations("见 refundPolicy.md", FINDINGS, KNOWN)
    assert r.ok and r.validity == 1.0 and footer(r) == ""


def test_fabricated_document_is_flagged_unknown():
    r = verify_citations("依据 hrHandbook.md 的规定", FINDINGS, KNOWN)
    assert r.unknown == ["hrHandbook.md"] and not r.ok and r.validity == 0.0


def test_real_document_never_retrieved_is_flagged_ungrounded():
    r = verify_citations("依据 salesCommissionRules.md", FINDINGS, KNOWN)      # exists, but not in this run's notes
    assert r.unknown == [] and r.ungrounded == ["salesCommissionRules.md"]


def test_partial_validity():
    r = verify_citations("refundPolicy.md 与 fake.md", FINDINGS, KNOWN)
    assert r.validity == 0.5 and r.cited == ["refundPolicy.md", "fake.md"]


def test_kb_unreachable_still_checks_groundedness():
    r = verify_citations("refundPolicy.md 与 salesCommissionRules.md", FINDINGS, None)
    assert r.unknown == [] and r.ungrounded == ["salesCommissionRules.md"]


@pytest.mark.parametrize("chinese,needle", [(True, "引用核验未通过"), (False, "Citation check failed")])
def test_footer_language(chinese, needle):
    r = verify_citations("fake.md", FINDINGS, KNOWN)
    assert needle in footer(r, chinese=chinese) and "fake.md" in footer(r, chinese=chinese)


def test_node_helper_uses_real_corpus_and_never_raises(monkeypatch):
    cfg = SimpleNamespace(postgres_connection_string="x")
    monkeypatch.setattr(dr, "load_corpus", lambda conn: SimpleNamespace(known_sources=KNOWN))
    res = asyncio.run(dr._check_citations("依据 fake.md", FINDINGS, cfg))
    assert res.unknown == ["fake.md"]
    assert asyncio.run(dr._check_citations("no citations at all", FINDINGS, cfg)) is None

    def boom(conn):
        raise RuntimeError("db down")

    monkeypatch.setattr(dr, "load_corpus", boom)
    res = asyncio.run(dr._check_citations("依据 salesCommissionRules.md", FINDINGS, cfg))
    assert res.ungrounded == ["salesCommissionRules.md"]          # degraded, not crashed
