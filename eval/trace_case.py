"""Run ONE case and print every tool call (name, args, first chars of the result) as it happens.

    python -m eval.trace_case refuse_drop

Use it to see *why* an agent run is slow or loops - the aggregate metrics only say that it is.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

from eval.cases import ALL_CASES
from eval.run_agent_eval import _real_invoke


class Printer(BaseCallbackHandler):
    def __init__(self):
        self.t0 = time.perf_counter()
        self.names = {}

    def _t(self):
        return f"[{time.perf_counter() - self.t0:6.1f}s]"

    def on_tool_start(self, serialized, input_str, *, run_id, **kw):
        self.names[run_id] = (serialized or {}).get("name") or kw.get("name")
        print(f"{self._t()} TOOL  {self.names[run_id]:<22} {str(input_str)[:140]!r}", flush=True)

    def on_tool_end(self, output, *, run_id, **kw):
        text = getattr(output, "content", output)
        print(f"{self._t()}   ->  {str(text)[:170]!r}", flush=True)

    def on_tool_error(self, error, *, run_id, **kw):
        print(f"{self._t()}   !!  {type(error).__name__}: {str(error)[:150]}", flush=True)


async def main(case_id: str):
    load_dotenv()
    case = next(c for c in ALL_CASES if c["id"] == case_id)
    config = {"callbacks": [Printer()], "configurable": {"thread_id": f"trace-{case_id}", "allow_clarification": False,
                                                          "approval_mode": "auto_approve"}}
    out = await _real_invoke(case, config)
    print("\nFINAL ANSWER:\n", out[:600])


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    assert os.getenv("INTERNAL_DB_URL"), "set INTERNAL_DB_URL"
    asyncio.run(main(sys.argv[1]))
