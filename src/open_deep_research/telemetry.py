"""Per-run telemetry: tokens, model/tool calls, node timings - as a LangChain callback.

One ``UsageTracker`` is attached to a graph run (``config["callbacks"]``) and sees every nested
LLM call, tool call and LangGraph node, including those inside sub-graphs. It keeps counters for
the task summary and, optionally, appends one JSON line per finished span to a trace file, which
is enough to answer "why was this task slow / expensive / failing?" without any external service.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

# Tools whose result text is safe to show the person who asked (no business data in it).
LIVE_PREVIEW_TOOLS = frozenset({"propose_refund", "withdraw_refund"})

TOOL_ERROR_PREFIXES = (
    "SQL execution error", "Query rejected", "Python execution error", "Python execution blocked",
    "Error:", "Error executing tool", "[TOOL_UNAVAILABLE]", "Memory search error", "Memory save error",
    "Knowledge base search error",
)


def _args_for_display(name: str, input_str, inputs) -> str:
    """A short, readable form of a tool's arguments for the live view. The agent's own reasoning is not shown."""
    if name == "think_tool":
        return ""
    if isinstance(inputs, dict):
        if isinstance(inputs.get("query"), str):
            return inputs["query"][:300]
        if isinstance(inputs.get("code"), str):
            return inputs["code"][:300]
        return ", ".join(f"{k}={str(v)[:60]}" for k, v in inputs.items())[:300]
    return str(input_str)[:200]


def _usage_of(response) -> tuple[int, int, str]:
    llm_out = response.llm_output or {}
    usage = llm_out.get("token_usage") or {}
    model = llm_out.get("model_name", "") or llm_out.get("model", "")
    if not usage:
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                meta = getattr(msg, "usage_metadata", None) or {}
                if meta:
                    usage = {"prompt_tokens": meta.get("input_tokens", 0), "completion_tokens": meta.get("output_tokens", 0)}
                model = model or (getattr(msg, "response_metadata", None) or {}).get("model_name", "")
    tin = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    tout = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    return int(tin), int(tout), model or "unknown"


class UsageTracker(BaseCallbackHandler):
    """Collects usage for one run. Also provides the attribute API the eval harness relies on."""

    raise_error = False                      # telemetry must never break a task

    def __init__(self, trace_path: Optional[str] = None, task_id: str = "") -> None:
        self.task_id = task_id
        self.tool_calls: list[dict] = []     # {"tool", "ok", "seconds"}
        self.llm_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0})
        self.node_seconds: dict[str, float] = defaultdict(float)
        self._starts: dict[Any, tuple[str, str, float]] = {}
        self._t0 = time.perf_counter()
        # what a UI can show while the task runs; drained by the service (deque pops are thread-safe)
        self.live_events: deque = deque(maxlen=500)
        self._trace = Path(trace_path) if trace_path else None
        if self._trace:
            self._trace.parent.mkdir(parents=True, exist_ok=True)

    # ---- trace file
    def _emit(self, **rec) -> None:
        if not self._trace:
            return
        rec = {"t": round(time.perf_counter() - self._t0, 3), "task": self.task_id, **rec}
        try:
            with self._trace.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ---- tools
    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown"
        self._starts[run_id] = ("tool", name, time.perf_counter())
        self.live_events.append({"type": "tool_start", "name": name, "args": _args_for_display(name, input_str, kwargs.get("inputs"))})

    def on_tool_end(self, output, *, run_id, **kwargs):
        _, name, t0 = self._starts.pop(run_id, ("tool", "unknown", time.perf_counter()))
        text = getattr(output, "content", output)
        text = text if isinstance(text, str) else str(text)
        ok = not text.startswith(TOOL_ERROR_PREFIXES)
        secs = round(time.perf_counter() - t0, 3)
        self.tool_calls.append({"tool": name, "ok": ok, "seconds": secs})
        self._emit(type="tool", name=name, ok=ok, seconds=secs, preview=text[:120])
        live = {"type": "tool_end", "name": name, "ok": ok, "seconds": secs}
        if name in LIVE_PREVIEW_TOOLS:
            live["preview"] = text[:300]
        self.live_events.append(live)

    def on_tool_error(self, error, *, run_id, **kwargs):
        _, name, t0 = self._starts.pop(run_id, ("tool", "unknown", time.perf_counter()))
        secs = round(time.perf_counter() - t0, 3)
        self.tool_calls.append({"tool": name, "ok": False, "seconds": secs})
        self._emit(type="tool", name=name, ok=False, seconds=secs, error=f"{type(error).__name__}: {str(error)[:120]}")
        self.live_events.append({"type": "tool_end", "name": name, "ok": False, "seconds": secs})

    # ---- llm
    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        self.llm_calls += 1
        if run_id is not None:
            self._starts[run_id] = ("llm", "llm", time.perf_counter())

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self.llm_calls += 1
        if run_id is not None:
            self._starts[run_id] = ("llm", "llm", time.perf_counter())

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        tin, tout, model = _usage_of(response)
        self.tokens_in += tin
        self.tokens_out += tout
        m = self.by_model[model]
        m["calls"] += 1; m["tokens_in"] += tin; m["tokens_out"] += tout
        _, _, t0 = self._starts.pop(run_id, ("llm", "llm", time.perf_counter()))
        self._emit(type="llm", model=model, tokens_in=tin, tokens_out=tout, seconds=round(time.perf_counter() - t0, 3))
        self.live_events.append({"type": "usage", "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                                 "llm_calls": self.llm_calls})

    def on_llm_error(self, error, *, run_id=None, **kwargs):
        self._starts.pop(run_id, None)
        self._emit(type="llm", ok=False, error=f"{type(error).__name__}: {str(error)[:120]}")

    # ---- graph nodes (LangGraph tags each node run with metadata["langgraph_node"])
    def on_chain_start(self, serialized, inputs, *, run_id, name=None, metadata=None, **kwargs):
        node = (metadata or {}).get("langgraph_node")
        if node and name == node:
            self._starts[run_id] = ("node", node, time.perf_counter())

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        entry = self._starts.pop(run_id, None)
        if entry and entry[0] == "node":
            secs = time.perf_counter() - entry[2]
            self.node_seconds[entry[1]] += secs
            self._emit(type="node", name=entry[1], seconds=round(secs, 3))

    def on_chain_error(self, error, *, run_id, **kwargs):
        entry = self._starts.pop(run_id, None)
        if entry and entry[0] == "node":
            self._emit(type="node", name=entry[1], ok=False, error=f"{type(error).__name__}: {str(error)[:120]}",
                       seconds=round(time.perf_counter() - entry[2], 3))

    # ---- summary
    def summary(self) -> dict:
        by_tool: dict[str, dict] = {}
        for c in self.tool_calls:
            t = by_tool.setdefault(c["tool"], {"calls": 0, "errors": 0, "seconds": 0.0})
            t["calls"] += 1
            t["errors"] += 0 if c["ok"] else 1
            t["seconds"] = round(t["seconds"] + c["seconds"], 3)
        return {
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out, "llm_calls": self.llm_calls,
            "tool_calls": len(self.tool_calls), "tool_errors": sum(1 for c in self.tool_calls if not c["ok"]),
            "seconds": round(time.perf_counter() - self._t0, 2),
            "by_tool": by_tool, "by_model": {k: dict(v) for k, v in self.by_model.items()},
            "node_seconds": {k: round(v, 2) for k, v in self.node_seconds.items()},
        }


def estimate_cost(tokens_in: int, tokens_out: int, price_in_per_m: Optional[float], price_out_per_m: Optional[float]) -> Optional[float]:
    """USD estimate, or None when prices are not configured (we do not guess provider prices)."""
    if price_in_per_m is None or price_out_per_m is None:
        return None
    return round((tokens_in * price_in_per_m + tokens_out * price_out_per_m) / 1e6, 6)
