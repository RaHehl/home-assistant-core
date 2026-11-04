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


def _read_until_eof(fd: int) -> bytes:
    data = b""
    while True:
        readable, _, _ = select.select([fd], [], [], TIMEOUT)
        assert readable, "no output in time"
        if not (chunk := os.read(fd, 65536)):
            return data
        data += chunk


class _ZygoteHarness:
    """Drives ``python -m homeassistant_sandbox.zygote`` over its pipes."""

    def __init__(self, log_dir: Path, *args: str, env: dict[str, str] | None = None):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "homeassistant_sandbox.zygote", *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._log_dir = log_dir
        self._buffer = b""

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

    def fork(self, name: str, argv: list[str]) -> tuple[int, int]:
        """Fork a worker; return its pid and the read end of its output."""
        log_path = self._log_dir / name
        os.mkfifo(log_path)
        log_fd = os.open(log_path, os.O_RDONLY | os.O_NONBLOCK)
        assert self.proc.stdin is not None
        request = {"op": "fork", "name": name, "argv": argv, "log": str(log_path)}
        self.proc.stdin.write(json.dumps(request).encode() + b"\n")
        self.proc.stdin.flush()
        report = self.receive()
        assert report["op"] == "forked"
        return report["pid"], log_fd


@pytest.fixture
def zygote(tmp_path: Path) -> Iterator[_ZygoteHarness]:
    """A running zygote, killed on teardown."""
    harness = _ZygoteHarness(tmp_path, "--grace", "1")
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
    ("argv", "code", "output"),
    [
        pytest.param(
            ["--url", "unix:///unused"], 2, b"required: --name", id="argparse"
        ),
        pytest.param(
            ["--name", "t", "--log-level", "BOGUS"],
            1,
            b"Unknown level",
            id="worker_exception",
        ),
    ],
)
def test_fork_reports_exit_code_and_output(
    zygote: _ZygoteHarness, argv: list[str], code: int, output: bytes
) -> None:
    """A fork reports the exit code; worker output goes to its own FIFO."""
    pid, log_fd = zygote.fork("t", argv)

    assert zygote.receive() == {"op": "exit", "pid": pid, "code": code}
    assert output in _read_until_eof(log_fd)
    os.close(log_fd)


def test_forked_worker_dials_back_named_after_its_group(
    zygote: _ZygoteHarness, listener: tuple[socket.socket, str]
) -> None:
    """A forked worker is the zygote's child, runs the runtime and is named."""
    server, url = listener
    pid, log_fd = zygote.fork("custom", ["--name", "custom", "--url", url])
    connection, _ = server.accept()

    assert _parent_pid(pid) == zygote.proc.pid
    assert Path(f"/proc/{pid}/comm").read_text(encoding="ascii") == "sbx-custom\n"
    assert (
        Path(f"/proc/{zygote.proc.pid}/comm").read_text(encoding="ascii")
        == "sbx-zygote\n"
    )

    os.kill(pid, signal.SIGKILL)
    assert zygote.receive() == {"op": "exit", "pid": pid, "code": -signal.SIGKILL}
    connection.close()
    os.close(log_fd)


def test_stdin_close_stops_zygote_and_workers(
    zygote: _ZygoteHarness, listener: tuple[socket.socket, str]
) -> None:
    """Core going away stops the zygote and every worker it forked."""
    server, url = listener
    pid, log_fd = zygote.fork("t", ["--name", "t", "--url", url])
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
    os.close(log_fd)


def test_preload_starting_a_thread_is_refused(tmp_path: Path) -> None:
    """Forking a process with threads is unsafe, so the zygote refuses."""
    (tmp_path / "threaded_module.py").write_text(
        "import threading, time\n"
        "threading.Thread(target=time.sleep, args=(60,), daemon=True).start()\n"
    )
    env = os.environ | {"PYTHONPATH": str(tmp_path)}
    zygote = _ZygoteHarness(tmp_path, "--preload", "threaded_module", env=env)

    _, stderr = zygote.proc.communicate(timeout=TIMEOUT)

    assert zygote.proc.returncode == 1
    assert b"preloading started a thread" in stderr
