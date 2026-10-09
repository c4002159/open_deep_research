"""Offline tests for the evaluation framework (no LLM, no network)."""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from eval.cases import ALL_CASES
from eval.harness import (
    TraceHandler,
    aggregate,
    build_result,
    compare_runs,
    extract_numbers,
    number_present,
    score_text,
)

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------ dataset integrity
def test_case_ids_unique_and_well_formed():
    ids = [c["id"] for c in ALL_CASES]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 60
    for c in ALL_CASES:
        assert c["question"].strip() and c["checks"], c["id"]
        assert set(c["checks"]) <= {"numbers", "contains_all", "contains_any", "db_unchanged", "cited_docs_exist"}


def test_every_tag_has_cases():
    tags = {c["tag"] for c in ALL_CASES}
    assert {"simple", "join_group", "analysis", "kb", "sql_kb", "safety", "out_of_scope", "error_recovery"} <= tags


@pytest.mark.skipif(not (ROOT / "test.db").exists(), reason="test.db not present")
def test_fixtures_match_test_db():
    """The CSV fixtures the expectations are computed from must equal the real database."""
    con = sqlite3.connect(ROOT / "test.db")
    for table, key in (("sales", "id"), ("products", "product_line")):
        db = pd.read_sql(f"select * from {table} order by {key}", con)
        csv = pd.read_csv(ROOT / "eval" / "fixtures" / f"{table}.csv")
        pd.testing.assert_frame_equal(db, csv, check_dtype=False)


# ------------------------------------------------------------ scoring
def test_number_extraction_handles_commas_and_wan():
    assert 2345328.08 in extract_numbers("总销售额为 2,345,328.08 元")
    assert 2_345_000 in extract_numbers("约 234.5万") or any(abs(n - 2_345_000) < 1 for n in extract_numbers("约 234.5万"))


def test_substring_false_positive_is_fixed():
    # the old metric accepted "80" inside "1180"
    assert not score_text({"numbers": [80]}, "共有 1180 条记录")["numbers"]
    assert score_text({"numbers": [80]}, "共有 80 条记录")["numbers"]


def test_tolerance_rules():
    assert number_present(2345328.08, [2345328.0])       # rounded to integer
    assert not number_present(15, [16])                  # small integers are exact
    assert number_present(-0.11, [-0.11])


def test_string_checks_ignore_whitespace_and_commas():
    assert score_text({"contains_all": ["15分钟"]}, "一级故障 15 分钟内响应")["contains_all"]
    assert not score_text({"contains_any": ["财务经理"]}, "由销售经理批准")["contains_any"]


def test_db_unchanged_check_and_infra_error():
    case = {"id": "x", "tag": "safety", "checks": {"contains_any": ["无法"], "db_unchanged": True}}
    ok = build_result(case, "无法执行", TraceHandler(), 1.0, "aaa", "aaa")
    bad = build_result(case, "无法执行", TraceHandler(), 1.0, "aaa", "bbb")
    err = build_result(case, "", TraceHandler(), 1.0, "aaa", "aaa", error="Timeout")
    assert ok.passed and not bad.passed and not err.passed


def test_trace_classifies_tool_errors():
    from uuid import uuid4

    t = TraceHandler()
    for out in ("a | b\n1 | 2", "Query rejected by read-only guard: x", "Python execution blocked: y"):
        rid = uuid4()
        t.on_tool_start({"name": "sql_executor"}, "q", run_id=rid)
        t.on_tool_end(out, run_id=rid)
    assert [c["ok"] for c in t.tool_calls] == [True, False, False]


# ------------------------------------------------------------ aggregation / comparison
def _res(i, passed, lat=1.0, tokens=1000, calls=2, errs=0, tag="simple"):
    return {"id": i, "tag": tag, "passed": passed, "latency_s": lat, "tool_calls": calls, "tool_errors": errs,
            "llm_calls": 3, "tokens_in": tokens, "tokens_out": 0, "error": None}


def test_aggregate_metrics():
    agg = aggregate([_res("a", True, lat=1), _res("b", False, lat=9, errs=1), _res("c", True, lat=2), _res("d", True, lat=3)])
    o = agg["overall"]
    assert o["task_success_rate"] == 0.75
    assert o["tool_call_success_rate"] == 0.875
    assert o["latency_p95_s"] == 9


def test_compare_flags_regressions_and_threshold_breach():
    base = {"results": [_res(str(i), True) for i in range(10)]}
    same = compare_runs(base, base)
    assert same["ok"] and not same["regressions"]

    worse = {"results": [_res(str(i), i >= 2, lat=5.0) for i in range(10)]}
    r = compare_runs(base, worse)
    assert r["regressions"] == ["0", "1"] and not r["ok"]
    assert any("task success" in p for p in r["problems"]) and any("p95" in p for p in r["problems"])


# ------------------------------------------------------------ pipeline
def test_dry_run_pipeline(tmp_path, monkeypatch):
    import asyncio

    import eval.run_agent_eval as runner

    monkeypatch.setattr(runner, "RESULTS", tmp_path)

    class A:
        name, tags, ids, limit, concurrency, force, dry_run = "t", None, None, 12, 4, False, True

    assert asyncio.run(runner.main_async(A())) == 0
    data = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    assert len(data["results"]) == 12 and data["summary"]["overall"]["cases"] == 12
    assert data["meta"]["dry_run"] is True


def test_cited_docs_exist_check():
    ok = score_text({"cited_docs_exist": True}, "依据 refundPolicy.md 第2节")
    bad = score_text({"cited_docs_exist": True}, "依据 refundPolicy.md 和 invented.md")
    none = score_text({"cited_docs_exist": True}, "没有引用任何文档")
    assert ok["cited_docs_exist"] and not bad["cited_docs_exist"] and none["cited_docs_exist"]


def test_every_kb_case_checks_citations():
    kb = [c for c in ALL_CASES if c["tag"] == "kb"]
    assert kb and all(c["checks"].get("cited_docs_exist") for c in kb)


# ------------------------------------------------------------ provider outages must not look like wrong answers
def _args(name, **kw):
    class A:
        tags = ids = limit = None
        concurrency, force, dry_run = 1, False, True

    A.name = name
    for k, v in kw.items():
        setattr(A, k, v)
    return A


def test_runner_aborts_when_the_provider_is_down_and_records_nothing(tmp_path, monkeypatch):
    import asyncio

    import eval.run_agent_eval as runner

    monkeypatch.setattr(runner, "RESULTS", tmp_path)
    calls = []

    async def dead_provider(case, config):
        calls.append(case["id"])
        raise RuntimeError("APIStatusError: Error code: 402 - {'error': {'message': 'Insufficient Balance'}}")

    monkeypatch.setattr(runner, "_fake_invoke", dead_provider)
    asyncio.run(runner.main_async(_args("outage", limit=20)))
    assert len(calls) == runner.MAX_CONSECUTIVE_PROVIDER_FAILURES, "must stop after N consecutive provider failures"
    data = json.loads((tmp_path / "outage.json").read_text(encoding="utf-8"))
    assert data["results"] == [], "an outage is not a result: nothing may be recorded as pass/fail"


def test_resume_reruns_cases_that_died_of_infrastructure_errors(tmp_path, monkeypatch):
    import asyncio

    import eval.run_agent_eval as runner

    monkeypatch.setattr(runner, "RESULTS", tmp_path)
    first = ALL_CASES[0]["id"]
    second = ALL_CASES[1]["id"]
    seen = []
    real = runner._fake_invoke

    async def spy(case, config):
        seen.append(case["id"])
        return await real(case, config)

    monkeypatch.setattr(runner, "_fake_invoke", spy)
    (tmp_path / "r.json").write_text(json.dumps({"results": [
        {"id": first, "tag": "simple", "passed": True, "error": None, "latency_s": 1, "tool_calls": 0, "tool_errors": 0,
         "llm_calls": 0, "tokens_in": 0, "tokens_out": 0},
        {"id": second, "tag": "simple", "passed": False, "error": "APIStatusError: Error code: 402", "latency_s": 1,
         "tool_calls": 0, "tool_errors": 0, "llm_calls": 0, "tokens_in": 0, "tokens_out": 0},
    ]}), encoding="utf-8")
    asyncio.run(runner.main_async(_args("r", ids=[first, second])))
    assert seen == [second], "the finished case is cached, the infrastructure failure is retried"


def test_provider_error_detection_has_no_false_positives_on_ids():
    from eval.run_agent_eval import PROVIDER_FATAL

    assert PROVIDER_FATAL.search("APIStatusError: Error code: 402 - Insufficient Balance")
    assert PROVIDER_FATAL.search("AuthenticationError: Incorrect API key provided")
    assert not PROVIDER_FATAL.search("request_id: 4402abc-4012 took 1402 ms")        # digits inside ids/numbers
    assert not PROVIDER_FATAL.search("SQL execution error: no such column")


def test_success_rate_is_reported_excluding_infrastructure_failures():
    rows = [_res_row(True), _res_row(True), _res_row(False), _res_row(False, error="402")]
    o = aggregate(rows)["overall"]
    assert o["task_success_rate"] == 0.5 and o["infra_errors"] == 1 and o["success_rate_excl_infra"] == 0.667


def _res_row(passed, error=None):
    return {"id": "x", "tag": "t", "passed": passed, "latency_s": 1.0, "tool_calls": 0, "tool_errors": 0,
            "llm_calls": 0, "tokens_in": 0, "tokens_out": 0, "error": error}
