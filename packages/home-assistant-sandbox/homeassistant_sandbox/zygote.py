"""Zygote that forks sandbox workers from one preloaded interpreter.

Protocol, one JSON object per line:

Core -> zygote (stdin):
    {"op": "fork", "argv": [...]}              argv for ``__main__.main``
    {"op": "signal", "pid": 123, "signal": 15}

Zygote -> Core (stdout):
    {"op": "ready", "pid": 100}
    {"op": "forked", "pid": 123}               one per fork, in request order
    {"op": "error", "error": "..."}            fork failed, in request order
    {"op": "exit", "pid": 123, "code": 0}      negative code = killed by signal

Closing stdin stops the zygote and terminates its workers.
"""

import argparse
import contextlib
import gc
import importlib
import json
import os
import selectors
import signal
import sys
import time
import traceback
from typing import Any

# Pulls in the runtime and the Core modules every worker needs.
from homeassistant_sandbox import __main__ as worker

SHUTDOWN_GRACE = 10.0


class Zygote:
    """Single-threaded fork server without an event loop."""

    def __init__(self, *, grace: float = SHUTDOWN_GRACE) -> None:
        """Set up the SIGCHLD self-pipe."""
        self._grace = grace
        self._children: set[int] = set()
        self._stopping = False
        self._wake_r, self._wake_w = os.pipe()
        os.set_blocking(self._wake_r, False)
        os.set_blocking(self._wake_w, False)
        self._selector = selectors.DefaultSelector()
        signal.set_wakeup_fd(self._wake_w)
        # A Python handler is needed for SIGCHLD to hit the wakeup fd.
        signal.signal(signal.SIGCHLD, lambda *_: None)
        signal.signal(signal.SIGTERM, self._on_sigterm)
        # Core drives shutdown by closing stdin; a terminal Ctrl-C must not
        # kill the zygote before Core gracefully stops the workers.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def serve(self) -> int:
        """Serve fork requests until stdin closes or SIGTERM arrives."""
        self._selector.register(0, selectors.EVENT_READ)
        self._selector.register(self._wake_r, selectors.EVENT_READ)
        self._send({"op": "ready", "pid": os.getpid()})
        buffer = b""
        while not self._stopping:
            for key, _ in self._selector.select():
                if key.fd == self._wake_r:
                    self._drain_wakeup()
                    self._reap()
                    continue
                data = os.read(0, 65536)
                if not data:
                    self._stopping = True
                    break
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        self._handle(json.loads(line))
        self._stop_children()
        return 0

    def _handle(self, request: dict[str, Any]) -> None:
        if request["op"] == "fork":
            self._fork(request["argv"])
        elif request["op"] == "signal":
            # Only signal unreaped children, so a recycled pid is never hit.
            if request["pid"] in self._children:
                os.kill(request["pid"], request["signal"])

    def _fork(self, argv: list[str]) -> None:
        try:
            pid = os.fork()
        except OSError as err:
            self._send({"op": "error", "error": str(err)})
            return
        if pid == 0:
            self._run_child(argv)
        self._children.add(pid)
        self._send({"op": "forked", "pid": pid})

    def _run_child(self, argv: list[str]) -> None:
        """Become a sandbox worker. Never returns."""
        code = 1
        try:
            signal.set_wakeup_fd(-1)
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.default_int_handler)
            self._selector.close()
            os.close(self._wake_r)
            os.close(self._wake_w)
            devnull = os.open(os.devnull, os.O_RDONLY)
            os.dup2(devnull, 0)
            os.close(devnull)
            # stdout is the control pipe to Core; send stray prints to the log.
            os.dup2(2, 1)
            code = worker.main(argv)
        except SystemExit as err:
            code = err.code if isinstance(err.code, int) else 1
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)

    def _reap(self) -> None:
        while self._children:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                return
            self._report_exit(pid, status)

    def _stop_children(self) -> None:
        """SIGTERM every worker, SIGKILL whatever outlives the grace window."""
        self._selector.unregister(0)
        for pid in self._children:
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + self._grace
        while self._children and (remaining := deadline - time.monotonic()) > 0:
            if self._selector.select(remaining):
                self._drain_wakeup()
            self._reap()
        for pid in self._children:
            os.kill(pid, signal.SIGKILL)
        while self._children:
            self._report_exit(*os.waitpid(-1, 0))

    def _report_exit(self, pid: int, status: int) -> None:
        self._children.discard(pid)
        self._send(
            {"op": "exit", "pid": pid, "code": os.waitstatus_to_exitcode(status)}
        )

    def _drain_wakeup(self) -> None:
        while True:
            try:
                if not os.read(self._wake_r, 512):
                    return
            except BlockingIOError:
                return

    def _on_sigterm(self, _signum: int, _frame: object) -> None:
        self._stopping = True

    def _send(self, message: dict[str, Any]) -> None:
        # Core may already be gone during shutdown; nobody is left to tell.
        with contextlib.suppress(OSError):
            os.write(1, json.dumps(message).encode() + b"\n")


def main(argv: list[str] | None = None) -> int:
    """Preload worker modules, freeze them, then serve fork requests."""
    parser = argparse.ArgumentParser(prog="python -m homeassistant_sandbox.zygote")
    parser.add_argument(
        "--preload",
        action="append",
        default=[],
        help="Extra module to import before forking (repeatable).",
    )
    parser.add_argument(
        "--grace",
        type=float,
        default=SHUTDOWN_GRACE,
        help="Seconds workers get between SIGTERM and SIGKILL on shutdown.",
    )
    args = parser.parse_args(argv)
    for module in args.preload:
        importlib.import_module(module)
    # Move preloaded objects out of the GC's reach so collections in the
    # workers do not touch, and thereby copy, the shared pages.
    gc.collect()
    gc.freeze()
    return Zygote(grace=args.grace).serve()


if __name__ == "__main__":
    raise SystemExit(main())
