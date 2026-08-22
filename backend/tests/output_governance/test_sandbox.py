import os
import sys

import pytest

from app.output_governance.sandbox_runner import (
    extract_code_block,
    run_code,
    sandbox_llm_output,
)

# resource.setrlimit is POSIX-only. On Windows (this project's local dev
# platform) the sandbox falls back to wall-clock-timeout-only protection
# — the memory/CPU-rlimit tests below would either be meaningless (no
# limit applied) or, worse, actually let a memory bomb consume real host
# RAM. They're skipped here and verified for real in CI, which runs on
# Linux — same pattern as Phase 1's tesseract-dependent tests.
requires_resource_module = pytest.mark.skipif(
    sys.platform == "win32", reason="resource.setrlimit is POSIX-only; not testable on Windows"
)


def test_normal_code_runs_and_returns_correct_output():
    result = run_code("print(1 + 1)")

    assert result.stdout.strip() == "2"
    assert result.exit_code == 0
    assert result.timed_out is False


def test_infinite_loop_is_killed_by_timeout():
    result = run_code("while True:\n    pass", timeout_seconds=1.0)

    assert result.timed_out is True
    assert result.exit_code is None


@requires_resource_module
def test_memory_bomb_is_killed_by_rlimit():
    # A single ~500MB allocation attempt against a 100MB RLIMIT_AS cap
    # fails immediately with MemoryError — no need for a slow-growing
    # loop, and no risk of actually consuming host RAM if the limit
    # works as intended.
    result = run_code(
        "x = bytearray(500 * 1024 * 1024)\nprint('should not get here')",
        memory_limit_bytes=100 * 1024 * 1024,
        timeout_seconds=5.0,
    )

    assert result.timed_out is False
    assert result.exit_code != 0
    assert "MemoryError" in result.stderr
    assert "should not get here" not in result.stdout


@requires_resource_module
def test_cpu_bound_busy_loop_is_killed_by_cpu_rlimit_before_wall_clock_timeout():
    # A tight CPU-bound loop should hit RLIMIT_CPU (2s) well before the
    # generous wall-clock backstop (30s) — proving the rlimit is doing
    # real work, not just the timeout.
    result = run_code(
        "x = 0\nwhile True:\n    x += 1",
        cpu_limit_seconds=2,
        timeout_seconds=30.0,
    )

    assert result.timed_out is False  # killed by SIGXCPU, not the wall clock
    assert result.exit_code != 0


def test_network_access_attempt_fails():
    result = run_code(
        "import socket\n"
        "s = socket.socket()\n"
        "s.connect(('example.com', 80))\n"
        "print('connected!')"
    )

    assert "connected!" not in result.stdout
    assert "OSError" in result.stderr
    assert result.exit_code != 0


def test_urllib_network_attempt_also_fails():
    # urllib builds on socket internally — the same monkeypatch should
    # stop it too, not just direct socket.socket() usage.
    result = run_code(
        "import urllib.request\n"
        "urllib.request.urlopen('http://example.com', timeout=2)\n"
        "print('connected!')"
    )

    assert "connected!" not in result.stdout
    assert result.exit_code != 0


def test_sandboxed_code_cannot_read_parent_process_environment():
    os.environ["SUPER_SECRET_TEST_VALUE"] = "leak-me-if-you-can"
    try:
        result = run_code(
            "import os\nprint(os.environ.get('SUPER_SECRET_TEST_VALUE', 'NOT_LEAKED'))"
        )
    finally:
        del os.environ["SUPER_SECRET_TEST_VALUE"]

    assert result.stdout.strip() == "NOT_LEAKED"


def test_code_that_raises_reports_nonzero_exit_and_stderr():
    result = run_code("raise ValueError('boom')")

    assert result.exit_code != 0
    assert "ValueError" in result.stderr
    assert "boom" in result.stderr


# --- extract_code_block / sandbox_llm_output ---


def test_extract_code_block_finds_fenced_python_block():
    text = "Here is code:\n```python\nprint(42)\n```\nHope that helps."

    code = extract_code_block(text)

    assert code == "print(42)\n"


def test_extract_code_block_works_without_language_tag():
    text = "```\nprint('hi')\n```"

    code = extract_code_block(text)

    assert code == "print('hi')\n"


def test_extract_code_block_returns_none_when_absent():
    assert extract_code_block("Just a normal sentence, no code here.") is None


def test_sandbox_llm_output_returns_none_for_plain_text():
    assert sandbox_llm_output("Here's a plain-text answer with no code.") is None


def test_sandbox_llm_output_runs_the_extracted_block():
    text = "Sure, here you go:\n```python\nprint('sandboxed')\n```"

    result = sandbox_llm_output(text)

    assert result is not None
    assert result.stdout.strip() == "sandboxed"
