"""Compare memory and startup of spawned vs zygote-forked sandbox workers.

Starts N workers through Core's SandboxManager (unix transport), measures
them idle, then sets up one real ``nam`` (Nettigo Air Monitor) config entry
per worker through the sandbox entry_setup path against a fake device
served from the test fixtures, and measures again.

Per worker: private memory (USS). Total: the workers' USS plus, for the
zygote modes, the zygote's anonymous memory once, so pages shared between
zygote and workers count once. Library pages also mapped by Core are left
out in every mode, and Core itself is excluded.

Workers lack the network integration that the shared aiohttp resolver
needs (KeyError 'network'), so a startup hook gives them a resolver with
their own zeroconf instead. It applies to every mode alike.

Linux only, needs ``nettigo-air-monitor``. Run from the repository root:

    python packages/home-assistant-sandbox/tools/measure_zygote.py -n 1 2 10
"""

# Stand-alone CLI driver, not a package module.
# ruff: noqa: INP001, T201, SLF001

import argparse
import asyncio
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

from aiohttp import web
from pytest_homeassistant.common import MockConfigEntry, async_test_home_assistant

from homeassistant.components.sandbox.bridge import SandboxBridge
from homeassistant.components.sandbox.channel import Channel
from homeassistant.components.sandbox.manager import (
    TRANSPORT_UNIX,
    SandboxConfig,
    SandboxManager,
)
from homeassistant.components.sandbox.messages import (
    MSG_ENTRY_SETUP,
    entry_to_setup_proto,
)
from homeassistant.components.sandbox.zygote import PRELOAD, ZygoteClient

NAM_DATA = (
    Path(__file__).resolve().parents[3] / "tests/components/nam/fixtures/nam_data.json"
).read_text()
MODES = {"spawn": None, "fork": PRELOAD, "fork-bare": ()}

# Patches aiohttp_client when it is first imported, so idle workers do not
# import it early.
RESOLVER_HOOK = """
import importlib.abc, importlib.util, sys

class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "homeassistant.helpers.aiohttp_client":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module
        def patched(module):
            exec_module(module)
            module._async_make_resolver = lambda hass: module.HassAsyncDNSResolver()
        spec.loader.exec_module = patched
        return spec

sys.meta_path.insert(0, _Hook())
"""


def _memory(pid: int) -> dict[str, float]:
    """Return smaps_rollup fields of ``pid`` in MB."""
    fields: dict[str, float] = {}
    rollup = Path(f"/proc/{pid}/smaps_rollup").read_text(encoding="ascii")
    for line in rollup.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == "kB":
            fields[parts[0].rstrip(":")] = int(parts[1]) / 1024
    fields["USS"] = fields["Private_Clean"] + fields["Private_Dirty"]
    return fields


async def _fake_nam() -> tuple[web.AppRunner, str]:
    """Serve the two endpoints the nam integration polls."""
    app = web.Application()
    app.router.add_get("/config", lambda _: web.Response(text="(aabbccddeeff)"))
    app.router.add_get(
        "/data.json",
        lambda _: web.Response(text=NAM_DATA, content_type="application/json"),
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, f"127.0.0.1:{port}"


def _report(
    manager: SandboxManager, mode: str, phase: str, pids: list[int]
) -> tuple[float, float]:
    usages = [_memory(pid)["USS"] for pid in pids]
    total = sum(usages)
    zygote = manager._zygote.pid if manager._zygote else None
    if zygote is not None:
        total += _memory(zygote)["Anonymous"]
    print(
        f"{mode:>10} {len(pids):>3} {phase:>5} {statistics.mean(usages):>9.1f}"
        f" {total:>9.1f} {total / len(pids):>9.1f}"
    )
    return statistics.mean(usages), total


async def _measure(mode: str, workers: int, settle: float) -> None:
    with tempfile.TemporaryDirectory() as config_dir:
        async with async_test_home_assistant(config_dir=config_dir) as hass:
            runner, host = await _fake_nam()
            bridges: dict[str, SandboxBridge] = {}

            def on_channel(group: str, channel: Channel) -> None:
                bridges[group] = SandboxBridge(hass, group=group, channel=channel)

            manager = SandboxManager(
                hass,
                config=SandboxConfig(ready_timeout=120.0),
                transport=TRANSPORT_UNIX,
                zygote=mode != "spawn",
                on_channel_ready=on_channel,
            )
            if (preload := MODES[mode]) is not None:
                manager._zygote = ZygoteClient(preload)
            try:
                pids: list[int] = []
                startup: list[float] = []
                for index in range(workers):
                    started = time.monotonic()
                    sandbox = await manager.ensure_started(f"w{index}")
                    startup.append(time.monotonic() - started)
                    assert sandbox.pid is not None
                    pids.append(sandbox.pid)
                await asyncio.sleep(settle)
                _report(manager, mode, "idle", pids)

                for index in range(workers):
                    group = f"w{index}"
                    entry = MockConfigEntry(
                        domain="nam",
                        sandbox=group,
                        title=f"NAM {index}",
                        unique_id=f"aa:bb:cc:dd:ee:{index:02x}",
                        data={"host": host},
                    )
                    entry.add_to_hass(hass)
                    bridges[group].entry_sync.track(entry)
                    request = entry_to_setup_proto(entry)
                    request.integration_source.kind = "builtin"
                    channel = manager.get(group).channel  # type: ignore[union-attr]
                    assert channel is not None
                    result = await channel.call(MSG_ENTRY_SETUP, request, timeout=120)
                    assert result.ok, result.reason
                async with asyncio.timeout(60):
                    while len(hass.states.async_entity_ids("sensor")) < workers:
                        await asyncio.sleep(0.1)
                await asyncio.sleep(settle)
                _report(manager, mode, "nam", pids)
                later = startup[1:] or startup
                print(
                    f"{'':>20} start: first {startup[0]:.2f} s,"
                    f" others {statistics.mean(later):.2f} s"
                )
            finally:
                await manager.async_stop_all()
                await runner.cleanup()


def _zygote_boot(preload: tuple[str, ...]) -> float:
    """Seconds from exec until the zygote reports ready."""
    command = [sys.executable, "-m", "homeassistant_sandbox.zygote"]
    for module in preload:
        command += ["--preload", module]
    started = time.monotonic()
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    ) as proc:
        assert proc.stdout is not None
        proc.stdout.readline()
        elapsed = time.monotonic() - started
        proc.communicate()
    return elapsed


def main() -> None:
    """Run every mode for every worker count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--workers", type=int, nargs="+", default=[1, 2, 10])
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument(
        "--settle", type=float, default=2.0, help="Seconds to idle before reading."
    )
    args = parser.parse_args()
    hook_dir = tempfile.mkdtemp(prefix="measure_zygote_")
    Path(hook_dir, "sitecustomize.py").write_text(RESOLVER_HOOK)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, (hook_dir, os.environ.get("PYTHONPATH")))
    )
    for name, preload in (("bare", ()), ("preload", PRELOAD)):
        boots = [_zygote_boot(preload) for _ in range(3)]
        print(f"zygote boot ({name}): {statistics.median(boots):.2f} s")
    print(
        f"{'mode':>10} {'n':>3} {'phase':>5} {'USS/wkr':>9} {'total':>9}"
        f" {'tot/wkr':>9}  (MB)"
    )
    for workers in args.workers:
        for mode in args.modes:
            asyncio.run(_measure(mode, workers, args.settle))


if __name__ == "__main__":
    main()
