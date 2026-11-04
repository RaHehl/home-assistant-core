"""Sandbox — fork workers from a preloaded zygote process."""

import asyncio
from collections import deque
import contextlib
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
from typing import Any

_LOGGER = logging.getLogger(__name__)

# Imported once in the zygote instead of in every worker that needs them.
# Measured per forked worker running the nam integration (measure_zygote.py):
PRELOAD = (
    # Saves 44 MB: pulls in hass_nabucasa, botocore, zeroconf and requests.
    "homeassistant.helpers.aiohttp_client",
    # Saves 2.5 MB alone, nothing on top of aiohttp_client; kept for
    # coordinator integrations that bring their own HTTP stack.
    "homeassistant.helpers.update_coordinator",
)


class ZygoteError(Exception):
    """The zygote could not fork a worker."""


class ZygoteProcess:
    """Process-like handle for a worker forked by the zygote.

    Quacks like ``asyncio.subprocess.Process`` for the manager. The worker
    is the zygote's child, so its exit status arrives as a zygote report.
    """

    stdin: asyncio.StreamWriter | None = None
    stdout: asyncio.StreamReader | None = None

    def __init__(self, pid: int, pidfd: int | None) -> None:
        """Initialise the handle."""
        self.pid = pid
        self.returncode: int | None = None
        self.stderr: asyncio.StreamReader | None = None
        self._pidfd = pidfd
        self._exited = asyncio.Event()

    async def wait(self) -> int:
        """Wait for the zygote to report the worker's exit."""
        await self._exited.wait()
        assert self.returncode is not None
        return self.returncode

    def send_signal(self, sig: int) -> None:
        """Signal the worker; a pidfd cannot hit a recycled pid."""
        if self._pidfd is not None:
            with contextlib.suppress(ProcessLookupError):
                signal.pidfd_send_signal(self._pidfd, sig)

    def terminate(self) -> None:
        """Send SIGTERM."""
        self.send_signal(signal.SIGTERM)

    def kill(self) -> None:
        """Send SIGKILL."""
        self.send_signal(signal.SIGKILL)

    def set_exited(self, code: int) -> None:
        """Record the worker's exit code."""
        self.returncode = code
        if self._pidfd is not None:
            os.close(self._pidfd)
            self._pidfd = None
        self._exited.set()


@dataclass
class _PendingFork:
    future: asyncio.Future[ZygoteProcess]
    log_fd: int


class ZygoteClient:
    """Starts the zygote on demand and forks workers from it."""

    def __init__(self, preload: tuple[str, ...] = PRELOAD) -> None:
        """Initialise the client; the zygote starts on the first fork."""
        self._command = [sys.executable, "-m", "homeassistant_sandbox.zygote"]
        for module in preload:
            self._command += ["--preload", module]
        self._proc: asyncio.subprocess.Process | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._start_lock = asyncio.Lock()
        self._pending: deque[_PendingFork] = deque()
        self._children: dict[int, ZygoteProcess] = {}

    @property
    def pid(self) -> int | None:
        """PID of the running zygote."""
        proc = self._proc
        return proc.pid if proc is not None and proc.returncode is None else None

    async def fork(
        self, name: str, argv: list[str], *, timeout: float
    ) -> ZygoteProcess:
        """Fork a worker running ``homeassistant_sandbox.__main__.main(argv)``."""
        async with self._start_lock:
            # Cleared only once the old zygote's reports are settled.
            if self._proc is None:
                await asyncio.wait_for(self._start(), timeout)
        log_dir = tempfile.mkdtemp(prefix=f"sandbox_{name}_log_")
        log_path = os.path.join(log_dir, "output")
        os.mkfifo(log_path, 0o600)
        log_fd = os.open(log_path, os.O_RDONLY | os.O_NONBLOCK)
        pending = _PendingFork(asyncio.get_running_loop().create_future(), log_fd)
        self._pending.append(pending)
        self._send({"op": "fork", "name": name, "argv": argv, "log": log_path})
        try:
            child = await asyncio.wait_for(pending.future, timeout)
        except BaseException:
            os.close(log_fd)
            raise
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)
        child.stderr = await self._open_log(log_fd)
        return child

    async def async_close(self) -> None:
        """Close the zygote's stdin and wait for it to stop its workers."""
        proc = self._proc
        if proc is not None and proc.stdin is not None:
            proc.stdin.close()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _send(self, message: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdin.is_closing():
            return
        proc.stdin.write(json.dumps(message).encode() + b"\n")

    async def _start(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdout is not None
        try:
            line = await proc.stdout.readline()
        except asyncio.CancelledError:
            proc.kill()
            raise
        if not line or json.loads(line)["op"] != "ready":
            _, stderr = await proc.communicate()
            raise ZygoteError(
                f"zygote failed during preload: {stderr.decode(errors='replace')}"
            )
        self._proc = proc
        self._tasks = [
            *(task for task in self._tasks if not task.done()),
            asyncio.create_task(self._read_reports(proc), name="sandbox zygote"),
            asyncio.create_task(self._drain_stderr(proc), name="sandbox zygote log"),
        ]

    async def _open_log(self, log_fd: int) -> asyncio.StreamReader:
        """Wrap the worker's output FIFO; EOF comes when the worker exits."""
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(log_fd, "rb")
        )
        return reader

    async def _read_reports(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        try:
            while line := await proc.stdout.readline():
                self._handle(proc.pid, json.loads(line))
        finally:
            await proc.wait()
            self._on_zygote_exit()

    def _handle(self, zygote_pid: int, report: dict[str, Any]) -> None:
        if report["op"] == "forked":
            child = ZygoteProcess(report["pid"], _open_pidfd(report["pid"], zygote_pid))
            self._children[child.pid] = child
            pending = self._pending.popleft()
            if not pending.future.done():
                pending.future.set_result(child)
            else:
                # The caller timed out; never leave the worker unsupervised.
                child.kill()
        elif report["op"] == "error":
            pending = self._pending.popleft()
            if not pending.future.done():
                pending.future.set_exception(ZygoteError(report["error"]))
        elif report["op"] == "exit":
            self._children.pop(report["pid"]).set_exited(report["code"])

    def _on_zygote_exit(self) -> None:
        """Fail pending forks and settle workers the zygote can no longer report."""
        _LOGGER.debug("Sandbox zygote exited")
        self._proc = None
        while self._pending:
            pending = self._pending.popleft()
            if not pending.future.done():
                pending.future.set_exception(ZygoteError("zygote exited"))
        for child in self._children.values():
            # Orphans of a killed zygote; kill them so a restart does not
            # leave a second worker for the same group running.
            child.kill()
            child.set_exited(-signal.SIGKILL)
        self._children.clear()

    async def _drain_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None
        while line := await proc.stderr.readline():
            if text := line.decode("utf-8", errors="replace").rstrip():
                _LOGGER.debug("sandbox zygote stderr: %s", text)


def _open_pidfd(pid: int, zygote_pid: int) -> int | None:
    """Return a pidfd for the zygote's child ``pid``, or None if it is gone."""
    try:
        pidfd = os.pidfd_open(pid)
    except ProcessLookupError:
        return None
    # The zygote may already have reaped the worker; then ``pid`` could
    # belong to an unrelated process.
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
    except FileNotFoundError:
        status = ""
    if f"\nPPid:\t{zygote_pid}\n" not in status:
        os.close(pidfd)
        return None
    return pidfd
