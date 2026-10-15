"""The web client: served by the service, no inline script, and model output can never inject markup."""

import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from open_deep_research.service.app import STATIC_DIR, create_app
from open_deep_research.service.settings import Settings

JS = (STATIC_DIR / "app.js").read_text(encoding="utf-8")


def test_the_three_client_files_exist_and_are_small_enough_to_read():
    for name, limit in (("index.html", 120), ("style.css", 400), ("app.js", 700)):
        lines = (STATIC_DIR / name).read_text(encoding="utf-8").count("\n")
        assert 0 < lines <= limit, f"{name} has {lines} lines"


def test_page_is_served_without_inline_script_and_the_files_resolve():
    async def go():
        app = create_app(Settings(api_keys={"k": "u"}), provider=type("P", (), {"graph": None, "start": lambda s: asyncio.sleep(0),
                                                                              "stop": lambda s: asyncio.sleep(0)})(), start_workers=False)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                page = await c.get("/")
                assert page.status_code == 200 and "text/html" in page.headers["content-type"]
                assert '<script src="/static/app.js"></script>' in page.text
                assert not re.search(r"<script(?![^>]*\bsrc=)", page.text), "inline scripts would defeat a strict CSP"
                assert (await c.get("/static/app.js")).status_code == 200 and (await c.get("/static/style.css")).status_code == 200
                assert (await c.get("/static/../app.py")).status_code in (404, 400)

    asyncio.run(go())


def test_there_is_exactly_one_html_sink_and_it_is_fed_by_the_escaping_renderer():
    sinks = re.findall(r"innerHTML|insertAdjacentHTML|document\.write|outerHTML", JS)
    assert sinks == ["innerHTML"], sinks
    assert "turn.report.innerHTML = md(" in JS
    assert "eval(" not in JS and "new Function" not in JS


node = shutil.which("node")


@pytest.mark.skipif(node is None, reason="node is not installed")
def test_markdown_renderer_escapes_before_it_adds_structure():
    start, end = JS.index("function md(src)"), JS.index("/* ------------------------------------------------------------------ banners")
    script = JS[start:end] + "\nconst cases = JSON.parse(process.argv[1]); console.log(JSON.stringify(cases.map(md)));"
    cases = ["<script>alert(1)</script> and <img src=x onerror=alert(1)>", "[click](javascript:alert(1)) [ok](https://example.com/a?b=1&c=2)",
             "**bold** and `<b>code</b>`", "```\n<script>x</script>\n```", "| a | b |\n|---|---|\n| <i>1</i> | 2 |", "# T\n- one\n- two",
             "1. first\n2. second"]
    out = json.loads(subprocess.run([node, "-e", script, json.dumps(cases)], capture_output=True, text=True, check=True, timeout=30).stdout)
    assert "<script" not in out[0] and "<img" not in out[0] and "&lt;script&gt;" in out[0]
    assert "href=\"javascript" not in out[1] and 'href="https://example.com/a?b=1&amp;c=2"' in out[1] and 'rel="noopener noreferrer"' in out[1]
    assert "<strong>bold</strong>" in out[2] and "<code>&lt;b&gt;code&lt;/b&gt;</code>" in out[2]
    assert "<script" not in out[3] and "&lt;script&gt;" in out[3]
    assert "<table>" in out[4] and "<i>" not in out[4] and "&lt;i&gt;" in out[4]
    assert "<h1>T</h1>" in out[5] and out[5].count("<li>") == 2 and "<ol>" in out[6]
