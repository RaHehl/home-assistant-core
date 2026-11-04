"""Sandbox workers forked from the zygote."""

import asyncio
from collections.abc import AsyncIterator
import os
from pathlib import Path
import signal

import pytest

from homeassistant.components.sandbox.manager import (
    TRANSPORT_STDIO,
    TRANSPORT_UNIX,
    SandboxConfig,
    SandboxManager,
    SandboxStartError,
)
from homeassistant.core import HomeAssistant

FAST_CONFIG = SandboxConfig(
    restart_limit=3,
    restart_window=60.0,
    restart_backoff=0.05,
    ready_timeout=20.0,
    shutdown_grace=5.0,
)


def _parent_pid(pid: int) -> int:
    status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
    line = next(line for line in status.splitlines() if line.startswith("PPid:"))
    return int(line.split()[1])


async def _wait_for_new_pid(manager: SandboxManager, group: str, old: int) -> int:
    sandbox = manager.get(group)
    assert sandbox is not None
    async with asyncio.timeout(FAST_CONFIG.ready_timeout):
        while sandbox.state != "running" or sandbox.pid in (None, old):
            await asyncio.sleep(0.05)
    assert sandbox.pid is not None
    return sandbox.pid


@pytest.fixture
async def manager(hass: HomeAssistant) -> AsyncIterator[SandboxManager]:
    """Zygote-mode manager; stops every worker and the zygote on teardown."""
    mgr = SandboxManager(
        hass, config=FAST_CONFIG, transport=TRANSPORT_UNIX, zygote=True
    )
    yield mgr
    await mgr.async_stop_all()


async def test_forked_workers_start_ping_and_stop(manager: SandboxManager) -> None:
    """Workers fork from one zygote, answer pings and shut down gracefully."""
    first = await manager.ensure_started("built-in")
    second = await manager.ensure_started("custom")
    zygote_pid = manager._zygote.pid  # type: ignore[union-attr]

    assert first.state == second.state == "running"
    assert first.pid is not None
    assert second.pid is not None
    assert _parent_pid(first.pid) == _parent_pid(second.pid) == zygote_pid
    assert first.channel is not None
    result = await asyncio.wait_for(first.channel.call("sandbox/ping", None), 5.0)
    assert result.pong == "sandbox"

    await manager.async_graceful_shutdown_all(timeout=FAST_CONFIG.shutdown_grace)
    await manager.async_stop_all()

    assert first.state == second.state == "stopped"
    assert manager._zygote.pid is None  # type: ignore[union-attr]
    with pytest.raises(ProcessLookupError):
        os.kill(zygote_pid, 0)


async def test_crash_restart_budget(hass: HomeAssistant) -> None:
    """A worker crashing on every fork exhausts the budget; the zygote survives."""
    forks: list[str] = []

    def failing_args(group: str, url: str) -> list[str]:
        forks.append(group)
        return ["--name", group, "--url", url, "--log-level", "BOGUS"]

    mgr = SandboxManager(
        hass,
        command_factory=failing_args,
        config=FAST_CONFIG,
        transport=TRANSPORT_UNIX,
        zygote=True,
    )
    try:
        with pytest.raises(SandboxStartError):
            await mgr.ensure_started("built-in")
        assert len(forks) == FAST_CONFIG.restart_limit
        assert mgr.get("built-in").state == "failed"  # type: ignore[union-attr]
        assert mgr._zygote.pid is not None  # type: ignore[union-attr]
    finally:
        await mgr.async_stop_all()


async def test_killed_worker_restarts(manager: SandboxManager) -> None:
    """A worker killed behind Core's back is reaped and forked again."""
    sandbox = await manager.ensure_started("built-in")
    assert sandbox.pid is not None
    old_pid = sandbox.pid

    os.kill(old_pid, signal.SIGKILL)

    await _wait_for_new_pid(manager, "built-in", old_pid)


async def test_killed_zygote_restarts_with_worker(manager: SandboxManager) -> None:
    """Losing the zygote kills its workers and the next fork restarts it."""
    sandbox = await manager.ensure_started("built-in")
    assert sandbox.pid is not None
    old_worker = sandbox.pid
    old_zygote = manager._zygote.pid  # type: ignore[union-attr]
    assert old_zygote is not None

    os.kill(old_zygote, signal.SIGKILL)

    new_worker = await _wait_for_new_pid(manager, "built-in", old_worker)
    new_zygote = manager._zygote.pid  # type: ignore[union-attr]
    assert new_zygote not in (None, old_zygote)
    assert _parent_pid(new_worker) == new_zygote
    with pytest.raises(ProcessLookupError):
        os.kill(old_worker, 0)


async def test_zygote_requires_unix_transport(hass: HomeAssistant) -> None:
    """Forked workers have no stdio pipes to Core."""
    with pytest.raises(ValueError, match="requires the unix transport"):
        SandboxManager(hass, transport=TRANSPORT_STDIO, zygote=True)
