"""Compare memory and startup of spawned vs zygote-forked sandbox workers.

Starts N idle workers through Core's SandboxManager (unix transport) both
ways and prints RSS/PSS/USS per worker from /proc/<pid>/smaps_rollup.
Linux only. Run from the repository root:

    python packages/home-assistant-sandbox/tools/measure_zygote.py -n 5
"""

# Stand-alone CLI driver, not a package module.
# ruff: noqa: INP001, T201

import argparse
import asyncio
from pathlib import Path
import statistics
import tempfile
import time

from homeassistant.components.sandbox.manager import (
    TRANSPORT_UNIX,
    SandboxConfig,
    SandboxManager,
)
from homeassistant.core import HomeAssistant


def _memory(pid: int) -> dict[str, float]:
    """Return RSS, PSS and USS of ``pid`` in MB."""
    fields: dict[str, int] = {}
    rollup = Path(f"/proc/{pid}/smaps_rollup").read_text(encoding="ascii")
    for line in rollup.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == "kB":
            fields[parts[0].rstrip(":")] = int(parts[1])
    return {
        "rss": fields["Rss"] / 1024,
        "pss": fields["Pss"] / 1024,
        "uss": (fields["Private_Clean"] + fields["Private_Dirty"]) / 1024,
    }


async def _measure(mode: str, workers: int, settle: float) -> None:
    with tempfile.TemporaryDirectory() as config_dir:
        hass = HomeAssistant(config_dir)
        manager = SandboxManager(
            hass,
            config=SandboxConfig(ready_timeout=60.0),
            transport=TRANSPORT_UNIX,
            zygote=mode == "zygote",
        )
        try:
            pids: list[int] = []
            startup: list[float] = []
            for index in range(workers):
                started = time.monotonic()
                sandbox = await manager.ensure_started(f"worker{index}")
                startup.append(time.monotonic() - started)
                assert sandbox.pid is not None
                pids.append(sandbox.pid)
            await asyncio.sleep(settle)

            print(f"\n{mode}: {workers} idle workers")
            print(
                f"{'pid':>8} {'start s':>8} {'RSS MB':>8} {'PSS MB':>8} {'USS MB':>8}"
            )
            usages = [_memory(pid) for pid in pids]
            for pid, seconds, usage in zip(pids, startup, usages, strict=True):
                print(
                    f"{pid:>8} {seconds:>8.2f} {usage['rss']:>8.1f}"
                    f" {usage['pss']:>8.1f} {usage['uss']:>8.1f}"
                )
            total_pss = sum(usage["pss"] for usage in usages)
            zygote_pid = manager._zygote.pid if manager._zygote else None  # noqa: SLF001
            if zygote_pid is not None:
                zygote = _memory(zygote_pid)
                total_pss += zygote["pss"]
                print(
                    f"{'zygote':>8} {'':>8} {zygote['rss']:>8.1f}"
                    f" {zygote['pss']:>8.1f} {zygote['uss']:>8.1f}"
                )
            print(
                f"mean per worker: RSS {statistics.mean(u['rss'] for u in usages):.1f}"
                f" PSS {statistics.mean(u['pss'] for u in usages):.1f}"
                f" USS {statistics.mean(u['uss'] for u in usages):.1f} MB,"
                f" start {statistics.mean(startup):.2f} s"
                f" (first {startup[0]:.2f} s)"
            )
            print(f"total PSS incl. zygote: {total_pss:.1f} MB")
        finally:
            await manager.async_stop_all()


def main() -> None:
    """Run both modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--workers", type=int, default=5)
    parser.add_argument(
        "--settle", type=float, default=2.0, help="Seconds to idle before reading."
    )
    args = parser.parse_args()
    for mode in ("spawn", "zygote"):
        asyncio.run(_measure(mode, args.workers, args.settle))


if __name__ == "__main__":
    main()
