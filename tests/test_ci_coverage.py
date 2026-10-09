"""Every test file must be run by the CI workflow. Two test files were once left out silently for a whole
stage; this makes forgetting one a test failure instead."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "eval-smoke.yml"


def test_every_test_file_is_in_the_ci_workflow():
    listed = set(re.findall(r"tests/(test_[A-Za-z0-9_]+\.py)", WORKFLOW.read_text(encoding="utf-8")))
    present = {p.name for p in (ROOT / "tests").glob("test_*.py")}
    assert present - listed == set(), f"not run by CI: {sorted(present - listed)}"
    assert listed - present == set(), f"CI lists files that do not exist: {sorted(listed - present)}"


def test_workflow_runs_on_every_push_and_can_be_started_by_hand():
    text = WORKFLOW.read_text(encoding="utf-8")
    on_block = text.split("jobs:")[0]
    assert "pull_request:" in on_block and "push:" in on_block and "workflow_dispatch:" in on_block
    assert "branches:" not in on_block, "a push filter would stop feature branches from being checked"
