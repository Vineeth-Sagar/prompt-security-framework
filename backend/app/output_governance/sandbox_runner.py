"""Sandboxed execution for fenced code blocks found in LLM output.

Extracts the first fenced code block (```...```) from an LLM response
and, if present, runs it in a locked-down child process — never on the
host directly, never with the backend's own environment. Layers of
isolation, each with an honest note on what it actually stops:

1. **Hard wall-clock timeout** (`subprocess.communicate(timeout=...)`) —
   cross-platform, kills an infinite loop regardless of OS.
2. **CPU time + address-space rlimits** (`resource.setrlimit`, POSIX
   only) — kills a CPU-bound busy-loop faster than the wall-clock
   timeout would, and kills a memory bomb via MemoryError before it can
   exhaust host RAM. Unavailable on Windows (no `resource` module) — on
   Windows only the wall-clock timeout protects the host, which is
   strictly weaker. This matters here specifically: local dev on this
   project happens on Windows, but the Docker image and CI both run
   Linux, so the real protection is verified there, not locally — same
   situation as Phase 1's tesseract-dependent tests.
3. **Blocked networking** — the child process's entry script monkeypatches
   `socket.socket`/`socket.create_connection` to raise before the
   user's code runs. This is NOT real network-namespace isolation (a
   Docker `--network none` container, which the original spec offers as
   a stronger alternative not built here) — it stops straightforward
   `socket`/`urllib`/`http.client`-based network calls, but a
   sufficiently creative escape (a C extension making a raw syscall
   directly) would not be caught by a Python-level monkeypatch.
4. **Stripped environment** — the child gets a minimal explicit `env`
   (just PATH), not the backend process's real environment. Without
   this, sandboxed code could simply `print(os.environ)` and exfiltrate
   this process's real API keys through the LLM response it arrived in.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel

_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)

DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_CPU_LIMIT_SECONDS = 5
DEFAULT_MEMORY_LIMIT_BYTES = 100 * 1024 * 1024  # 100MB

_NETWORK_BLOCK_PREAMBLE = """\
import socket as _sandbox_socket


def _sandbox_block_network(*args, **kwargs):
    raise OSError("network access is disabled in this sandbox")


_sandbox_socket.socket = _sandbox_block_network
_sandbox_socket.create_connection = _sandbox_block_network
"""


class SandboxResult(BaseModel):
    """Outcome of running one code block in the sandbox.

    Attributes:
        stdout, stderr: Captured output, truncated to whatever the
            process produced before it exited, was killed by timeout,
            or hit a resource limit.
        exit_code: The process's exit code, or None if it was killed by
            the wall-clock timeout (no exit code exists in that case).
        timed_out: True if the wall-clock timeout fired.
        code: The exact source that was executed, for the
            explainability log.
    """

    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    code: str


def extract_code_block(text: str) -> str | None:
    """Return the first fenced code block's contents, or None if there isn't one."""
    match = _CODE_BLOCK_RE.search(text)
    return match.group(1) if match else None


def _resource_limit_preexec(cpu_limit_seconds: float, memory_limit_bytes: int):
    """Build a preexec_fn applying rlimits — POSIX only, None on Windows.

    Called in the child process after fork() but before exec(), which
    is the only place `resource.setrlimit` can actually constrain the
    process about to run untrusted code.
    """
    if sys.platform == "win32":
        return None

    import resource

    def _apply() -> None:
        cpu = int(cpu_limit_seconds)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))

    return _apply


def run_code(
    code: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    cpu_limit_seconds: float = DEFAULT_CPU_LIMIT_SECONDS,
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES,
) -> SandboxResult:
    """Execute `code` in a locked-down child process and capture the result.

    Never raises for the code under test misbehaving (infinite loop,
    crash, memory bomb, network attempt) — all of those come back as a
    normal SandboxResult with the relevant fields set, since that's
    exactly the information the caller needs to log/report.
    """
    full_source = _NETWORK_BLOCK_PREAMBLE + "\n" + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_source)
        script_path = f.name

    minimal_env = {"PATH": os.environ.get("PATH", "")}

    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            env=minimal_env,
            preexec_fn=_resource_limit_preexec(cpu_limit_seconds, memory_limit_bytes),
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                timed_out=False,
                code=code,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=None,
                timed_out=True,
                code=code,
            )
    finally:
        Path(script_path).unlink(missing_ok=True)


def sandbox_llm_output(text: str, **kwargs) -> SandboxResult | None:
    """Extract and run the first fenced code block in `text`, if any.

    Returns None (not an error) when `text` contains no fenced code
    block — most LLM responses don't, and that's the expected case, not
    a failure.
    """
    code = extract_code_block(text)
    if code is None:
        return None
    return run_code(code, **kwargs)
