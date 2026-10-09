"""Isolated execution for the agent's ``python_executor`` tool.

Layers (outermost first):

1. Static policy (``check_code_policy``): AST check for imports, dangerous
   builtins, dunder access and a few known escape routes. Gives the model a clear,
   fixable error and rejects obvious attacks without starting a process.
2. Process isolation: code runs in a separate interpreter with an empty
   environment (no API keys), a temp working directory, a wall-clock timeout and an
   output cap, using restricted builtins and an import allow-list.
3. Container isolation: the code runs inside llm-sandbox with no network, a memory
   cap, a CPU cap, a non-root user and no-new-privileges.

Layers 1-2 are defence in depth, NOT a security boundary: in-process Python restrictions
can be bypassed by a determined attacker. The container is the boundary, so it is
REQUIRED by default and the tool refuses to run code when no Docker daemon is reachable.
``CODE_SANDBOX_BACKEND`` selects the behaviour:

  docker      (default) container only; refuse to execute if Docker is unavailable
  auto        container if available, otherwise the subprocess fallback + a warning (development)
  subprocess  always the subprocess fallback (tests, machines without Docker; weaker isolation)
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

from open_deep_research.policy import fatal_denial

logger = logging.getLogger(__name__)

SENTINEL = "\x00__SANDBOX_RESULT__\x00"

# POSIX only. This caps *virtual address space*, not resident memory, so it is a coarse guard against a
# runaway allocation - the container's memory limit is the real one. It must not be tight: measured on Linux,
# `import scipy.stats` HANGS under a 1 GiB cap and imports in <1 s under 2 GiB, so 4 GiB leaves headroom.
SUBPROCESS_ADDRESS_SPACE_LIMIT = 4 << 30
MAX_OUTPUT_CHARS = 20_000

ALLOWED_IMPORTS = frozenset({
    "pandas", "numpy", "scipy", "math", "statistics", "json", "datetime", "re",
    "collections", "itertools", "functools", "decimal", "fractions", "random",
    "string", "textwrap", "typing", "dataclasses", "operator", "heapq", "bisect",
})

FORBIDDEN_CALLS = frozenset({
    "exec", "eval", "compile", "open", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "help",
    "exit", "quit", "memoryview",
})

# Attribute names that lead to os/sys/file/network access through pandas/numpy
# internals (e.g. ``pd.io.common.os.system``) or write/read files directly.
FORBIDDEN_ATTRS = frozenset({
    "os", "sys", "subprocess", "system", "popen", "builtins", "importlib",
    "io", "compat", "ctypes", "shutil", "socket", "testing", "loadtxt",
    "genfromtxt", "fromfile", "tofile", "load", "save", "savetxt", "memmap",
    "to_pickle", "to_csv", "to_excel", "to_parquet", "to_sql", "to_hdf",
    "to_clipboard", "to_feather", "to_json", "to_xml", "to_html", "to_latex",
})


DOCKER_RUNTIME_CONFIGS = {
    "network_mode": "none",
    "mem_limit": "256m",
    "nano_cpus": 1_000_000_000,
    "pids_limit": 64,
    "cap_drop": ["ALL"],
    "security_opt": ["no-new-privileges"],
    "user": "sandbox",  # llm-sandbox otherwise runs as root; the image defines this user
}


DANGEROUS_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "ctypes", "urllib", "http", "requests",
    "pickle", "importlib", "builtins", "multiprocessing", "threading", "signal", "pathlib",
})


class CodePolicyViolation(Exception):
    """Raised when code fails the static policy.

    ``fatal`` marks attempts at shell/file/network/introspection access (retrying is
    pointless); a merely unavailable library is fixable and not fatal.
    """

    def __init__(self, message: str, fatal: bool = True):
        super().__init__(message)
        self.fatal = fatal


def check_code_policy(code: str) -> None:
    """Raise ``CodePolicyViolation`` if ``code`` uses anything outside the policy."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return  # let the interpreter report the syntax error with a normal traceback

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise CodePolicyViolation("relative imports are not allowed")
            _check_module(node.module or "")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_CALLS:
                raise CodePolicyViolation(f"'{node.id}' is not allowed")
            if node.id.startswith("__"):
                raise CodePolicyViolation(f"access to '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise CodePolicyViolation(f"access to '{node.attr}' is not allowed")
            if node.attr in FORBIDDEN_ATTRS or node.attr.startswith("read_"):
                raise CodePolicyViolation(f"attribute '.{node.attr}' is not allowed")


def _check_module(name: str) -> None:
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise CodePolicyViolation(
            f"import of '{root}' is not allowed (allowed: {', '.join(sorted(ALLOWED_IMPORTS))})",
            fatal=root in DANGEROUS_MODULES,
        )


# Runs inside the child interpreter / container. ``_CODE`` is prepended by the caller
# (None => read the code from stdin).
_RUNNER = r'''
import sys, io, json, builtins, traceback
_real_out = sys.stdout
_LIMIT = %(limit)d

class _Capped(io.StringIO):
    def write(self, s):
        if self.tell() + len(s) > _LIMIT:
            raise RuntimeError("output limit exceeded")
        return super().write(s)

_ALLOWED = %(allowed)r
_DENIED = %(denied)r
_real_import = builtins.__import__

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in _ALLOWED:
        raise ImportError("import of %%r is not allowed" %% name)
    return _real_import(name, globals, locals, fromlist, level)

_safe = {k: v for k, v in vars(builtins).items() if k not in _DENIED}
_safe["__import__"] = _safe_import

_code = _CODE if _CODE is not None else sys.stdin.read()
_out = {"stdout": "", "result": None, "error": None}
try:
    import pandas as pd, numpy as np
    ns = {"__builtins__": _safe, "pd": pd, "np": np}
    if "scipy" in _code:  # scipy is slow to import; only pay for it when used
        import scipy
        ns["scipy"] = scipy
    cap = _Capped()
    sys.stdout = cap
    try:
        exec(compile(_code, "<agent_code>", "exec"), ns)
    finally:
        sys.stdout = _real_out
    _out["stdout"] = cap.getvalue()
    if ns.get("result") is not None:
        _out["result"] = repr(ns["result"])[:%(limit)d]
except BaseException:
    sys.stdout = _real_out
    _out["error"] = traceback.format_exc(limit=-3)[-4000:]
_real_out.write(%(sentinel)r + json.dumps(_out))
_real_out.flush()
'''


def _runner_source(code: Optional[str]) -> str:
    body = _RUNNER % {
        "limit": MAX_OUTPUT_CHARS,
        "allowed": set(ALLOWED_IMPORTS),
        "denied": set(FORBIDDEN_CALLS) | {"__loader__", "__spec__"},
        "sentinel": SENTINEL,
    }
    return f"_CODE = {code!r}\n" + body


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    result: Optional[str] = None
    error: Optional[str] = None
    backend: str = ""
    blocked: bool = False  # rejected by static policy / timeout / limits (not a user-code bug)
    fatal: bool = False  # blocked for a security reason; retrying cannot help


def _parse_output(raw: str, backend: str) -> SandboxResult:
    if SENTINEL not in raw:
        return SandboxResult(False, error=f"sandbox produced no result: {raw[-500:]}", backend=backend, blocked=True)
    payload = json.loads(raw.rsplit(SENTINEL, 1)[1])
    return SandboxResult(
        ok=payload["error"] is None,
        stdout=payload["stdout"],
        result=payload["result"],
        error=payload["error"],
        backend=backend,
    )


def _run_subprocess(code: str, timeout: float) -> SandboxResult:
    env = {"PYTHONIOENCODING": "utf-8"}
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    kwargs = {}
    if os.name == "posix":
        import resource

        def _limits():
            resource.setrlimit(resource.RLIMIT_AS, (SUBPROCESS_ADDRESS_SPACE_LIMIT, SUBPROCESS_ADDRESS_SPACE_LIMIT))
            resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 1))

        kwargs["preexec_fn"] = _limits

    with tempfile.TemporaryDirectory(prefix="agent_sbx_") as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _runner_source(None)],
                input=code.encode("utf-8"),
                capture_output=True,
                timeout=timeout,
                cwd=workdir,
                env=env,
                **kwargs,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(False, error=f"execution exceeded {timeout:.0f}s and was killed", backend="subprocess", blocked=True)
    out = proc.stdout.decode("utf-8", "replace")
    if SENTINEL not in out:
        if proc.returncode < 0:  # POSIX: killed by a signal - RLIMIT_CPU sends SIGXCPU, a runaway allocation gets SIGKILL
            return SandboxResult(False, error=f"execution was killed by the resource limits (signal {-proc.returncode}); "
                                              "it used too much CPU time or memory", backend="subprocess", blocked=True)
        return SandboxResult(False, error=f"sandbox crashed: {proc.stderr.decode('utf-8', 'replace')[-500:]}", backend="subprocess", blocked=True)
    return _parse_output(out, "subprocess")


def _run_docker(code: str, timeout: float) -> SandboxResult:
    from llm_sandbox import SandboxSession

    image = os.environ.get("CODE_SANDBOX_IMAGE", "agent-sandbox-python:latest")
    try:
        with SandboxSession(
            lang="python",
            image=image,
            skip_environment_setup=True,
            execution_timeout=timeout,
            runtime_configs=DOCKER_RUNTIME_CONFIGS,
        ) as session:
            res = session.run(_runner_source(code))
    except Exception as e:  # container start failure, timeout, OOM kill ...
        return SandboxResult(False, error=f"docker sandbox error: {str(e)[:300]}", backend="docker", blocked=True)
    return _parse_output(res.stdout or "", "docker") if SENTINEL in (res.stdout or "") else SandboxResult(
        False, error=f"sandbox crashed: {(res.stderr or '')[-500:]}", backend="docker", blocked=True
    )


_docker_cache: tuple[float, bool] = (0.0, False)


def docker_available() -> bool:
    global _docker_cache
    if time.monotonic() - _docker_cache[0] < 30 and _docker_cache[0]:
        return _docker_cache[1]
    try:
        import docker

        docker.from_env(timeout=2).ping()
        ok = True
    except Exception:
        ok = False
    _docker_cache = (time.monotonic(), ok)
    return ok


class SandboxUnavailable(RuntimeError):
    """The required isolation is not available; code must not be executed."""


def resolve_backend() -> str:
    choice = os.environ.get("CODE_SANDBOX_BACKEND", "docker").lower()
    if choice == "subprocess":
        return "subprocess"
    if docker_available():
        return "docker"
    if choice == "auto":
        logger.warning("Docker unavailable; falling back to the subprocess sandbox (weaker isolation)")
        return "subprocess"
    raise SandboxUnavailable(
        "the code sandbox needs a running Docker daemon (and the agent-sandbox-python image) and none is "
        "reachable, so no code was executed. Start Docker, or for local development only set "
        "CODE_SANDBOX_BACKEND=auto (falls back to a weaker subprocess sandbox). Do not retry")


def run_python_sandboxed_sync(code: str, timeout: float = 20.0, backend: Optional[str] = None) -> SandboxResult:
    try:
        check_code_policy(code)
    except CodePolicyViolation as e:
        return SandboxResult(False, error=f"Blocked by sandbox policy: {e}", backend="policy", blocked=True,
                             fatal=getattr(e, "fatal", True))
    try:
        backend = backend or resolve_backend()
    except SandboxUnavailable as e:
        return SandboxResult(False, error=str(e), backend="none", blocked=True)
    return _run_docker(code, timeout) if backend == "docker" else _run_subprocess(code, timeout)


async def run_python_sandboxed(code: str, timeout: float = 20.0) -> SandboxResult:
    return await asyncio.to_thread(run_python_sandboxed_sync, code, timeout)


def format_for_agent(res: SandboxResult) -> str:
    """Render a result in the same shape the old in-process executor returned."""
    if res.fatal:
        return fatal_denial(f"Python execution blocked: {res.error}")
    if res.blocked:
        return f"Python execution blocked: {res.error}"
    if not res.ok:
        return f"Python execution error:\n{res.error}"
    out = res.stdout
    if res.result is not None:
        out += f"\nresult = {res.result}"
    return out or (
        "Code executed successfully with no output. "
        "Did you forget to set `result` or print something?"
    )
