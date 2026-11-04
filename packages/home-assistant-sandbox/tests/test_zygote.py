"""Zygote fork server."""

from collections.abc import Iterator
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
from typing import Any

import pytest

TIMEOUT = 30.0


class _ZygoteHarness:
    """Drives ``python -m homeassistant_sandbox.zygote`` over its pipes."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "homeassistant_sandbox.zygote", "--grace", "1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self._buffer = b""

    def send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message).encode() + b"\n")
        self.proc.stdin.flush()

    def receive(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buffer:
            readable, _, _ = select.select([fd], [], [], TIMEOUT)
            assert readable, "zygote did not report in time"
            data = os.read(fd, 65536)
            assert data, "zygote closed stdout"
            self._buffer += data
        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line)

    def fork(self, argv: list[str]) -> int:
        self.send({"op": "fork", "argv": argv})
        report = self.receive()
        assert report["op"] == "forked"
        return report["pid"]


@pytest.fixture
def zygote() -> Iterator[_ZygoteHarness]:
    """A running zygote, killed on teardown."""
    harness = _ZygoteHarness()
    assert harness.receive() == {"op": "ready", "pid": harness.proc.pid}
    yield harness
    harness.proc.kill()
    harness.proc.wait()


@pytest.fixture
def listener() -> Iterator[tuple[socket.socket, str]]:
    """A unix socket standing in for Core's control-channel listener."""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "control.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen()
        server.settimeout(TIMEOUT)
        yield server, f"unix://{path}"
        server.close()


def _parent_pid(pid: int) -> int:
    status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
    line = next(line for line in status.splitlines() if line.startswith("PPid:"))
    return int(line.split()[1])


@pytest.mark.parametrize(
    ("argv", "code"),
    [
        pytest.param(["--url", "unix:///unused"], 2, id="argparse_error"),
        pytest.param(["--name", "t", "--log-level", "BOGUS"], 1, id="worker_exception"),
    ],
)
def test_fork_reports_pid_and_exit_code(
    zygote: _ZygoteHarness, argv: list[str], code: int
) -> None:
    """Each fork replies with the pid and later with the worker's exit code."""
    pid = zygote.fork(argv)

    assert zygote.receive() == {"op": "exit", "pid": pid, "code": code}


def test_forked_worker_dials_back_and_can_be_signalled(
    zygote: _ZygoteHarness, listener: tuple[socket.socket, str]
) -> None:
    """A forked worker is the zygote's child, runs the runtime and obeys signals."""
    server, url = listener
    pid = zygote.fork(["--name", "t", "--url", url])
    connection, _ = server.accept()

    assert _parent_pid(pid) == zygote.proc.pid

    zygote.send({"op": "signal", "pid": pid, "signal": signal.SIGKILL})
    assert zygote.receive() == {"op": "exit", "pid": pid, "code": -signal.SIGKILL}
    connection.close()


def test_stdin_close_stops_zygote_and_workers(
    zygote: _ZygoteHarness, listener: tuple[socket.socket, str]
) -> None:
    """Core going away stops the zygote and every worker it forked."""
    server, url = listener
    pid = zygote.fork(["--name", "t", "--url", url])
    connection, _ = server.accept()

    assert zygote.proc.stdin is not None
    zygote.proc.stdin.close()

    report = zygote.receive()
    assert report["op"] == "exit"
    assert report["pid"] == pid
    assert zygote.proc.wait(TIMEOUT) == 0
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    connection.close()
