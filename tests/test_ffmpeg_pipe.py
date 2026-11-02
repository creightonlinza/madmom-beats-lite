from __future__ import annotations

import subprocess
import sys

from madmom.io.audio import _pipe_through


def _python(code: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE)


def test_pipes_all_data_through_the_process() -> None:
    data = bytes(range(256)) * 40000
    proc = _python("import shutil, sys; shutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer)")
    assert _pipe_through(proc, memoryview(data)) == data
    assert proc.returncode == 0


def test_closes_stdin_when_the_process_quits_early() -> None:
    proc = _python("import sys; sys.exit(3)")
    assert _pipe_through(proc, memoryview(bytes(8_000_000))) == b""
    assert proc.returncode == 3
    assert proc.stdin.closed
