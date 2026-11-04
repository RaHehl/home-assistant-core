"""Sandbox — fork workers from a preloaded zygote process."""

import asyncio
from collections import deque
import contextlib
import json
import logging
import os
import signal
import sys
from typing import Any

_LOGGER = logging.getLogger(__name__)

ZYGOTE_COMMAND = [sys.executable, "-m", "homeassistant_sandbox.zygote"]


class ZygoteError(Exception):
    """The zygote could not fork a worker."""


class ZygoteProcess:
    """Process-like handle for a worker forked by the zygote.

    Quacks like ``asyncio.subprocess.Process`` for the manager. The worker
    is the zygote's child, so its exit status arrives as a zygote report.
    """

    stdin: asyncio.StreamWriter | None = None
    stdout: asyncio.StreamReader | None = None
    stderr: asyncio.StreamReader | None = None

    def __init__(self, client: ZygoteClient, pid: int) -> None:
        """Initialise the handle."""
        self._client = client
        self.pid = pid
        self.returncode: int | None = None
        self._exited = asyncio.Event()

    async def wait(self) -> int:
        """Wait for the zygote to report the worker's exit."""
        await self._exited.wait()
        assert self.returncode is not None
        return self.returncode

    def send_signal(self, sig: int) -> None:
        """Ask the zygote to signal the worker."""
        if self.returncode is None:
            self._client.send({"op": "signal", "pid": self.pid, "signal": sig})

    def terminate(self) -> None:
        """Send SIGTERM."""
        self.send_signal(signal.SIGTERM)

    def kill(self) -> None:
        """Send SIGKILL."""
        self.send_signal(signal.SIGKILL)

    def set_exited(self, code: int) -> None:
        """Record the worker's exit code."""
        self.returncode = code
        self._exited.set()


class ZygoteClient:
    """Starts the zygote on demand and forks workers from it."""

    def __init__(self, command: list[str] | None = None) -> None:
        """Initialise the client; the zygote starts on the first fork."""
        self._command = command or ZYGOTE_COMMAND
        self._proc: asyncio.subprocess.Process | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._start_lock = asyncio.Lock()
        self._pending: deque[asyncio.Future[ZygoteProcess]] = deque()
        self._children: dict[int, ZygoteProcess] = {}

    @property
    def pid(self) -> int | None:
        """PID of the running zygote."""
        proc = self._proc
        return proc.pid if proc is not None and proc.returncode is None else None

    async def fork(self, argv: list[str], *, timeout: float) -> ZygoteProcess:
        """Fork a worker running ``homeassistant_sandbox.__main__.main(argv)``."""
        async with self._start_lock:
            # Cleared only once the old zygote's reports are settled.
            if self._proc is None:
                await asyncio.wait_for(self._start(), timeout)
        future: asyncio.Future[ZygoteProcess] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending.append(future)
        self.send({"op": "fork", "argv": argv})
        return await asyncio.wait_for(future, timeout)

    def send(self, message: dict[str, Any]) -> None:
        """Write one request line to the zygote."""
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdin.is_closing():
            return
        proc.stdin.write(json.dumps(message).encode() + b"\n")

    async def async_close(self) -> None:
        """Close the zygote's stdin and wait for it to stop its workers."""
        proc = self._proc
        if proc is not None and proc.stdin is not None:
            proc.stdin.close()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

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

    async def _read_reports(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        try:
            while line := await proc.stdout.readline():
                self._handle(json.loads(line))
        finally:
            await proc.wait()
            self._on_zygote_exit()

    def _handle(self, report: dict[str, Any]) -> None:
        if report["op"] == "forked":
            child = ZygoteProcess(self, report["pid"])
            self._children[child.pid] = child
            future = self._pending.popleft()
            if not future.done():
                future.set_result(child)
            else:
                # The caller timed out; never leave the worker unsupervised.
                child.kill()
        elif report["op"] == "error":
            future = self._pending.popleft()
            if not future.done():
                future.set_exception(ZygoteError(report["error"]))
        elif report["op"] == "exit":
            self._children.pop(report["pid"]).set_exited(report["code"])

    def _on_zygote_exit(self) -> None:
        """Fail pending forks and settle workers the zygote can no longer report."""
        _LOGGER.debug("Sandbox zygote exited")
        self._proc = None
        while self._pending:
            future = self._pending.popleft()
            if not future.done():
                future.set_exception(ZygoteError("zygote exited"))
        for child in self._children.values():
            # Orphans of a killed zygote; kill them so a restart does not
            # leave a second worker for the same group running.
            with contextlib.suppress(ProcessLookupError):
                os.kill(child.pid, signal.SIGKILL)
            child.set_exited(-signal.SIGKILL)
        self._children.clear()

    async def _drain_stderr(self, proc: asyncio.subprocess.Process) -> None:
        """Log worker and zygote output; forked workers share this pipe."""
        assert proc.stderr is not None
        while line := await proc.stderr.readline():
            if text := line.decode("utf-8", errors="replace").rstrip():
                _LOGGER.debug("sandbox zygote stderr: %s", text)
