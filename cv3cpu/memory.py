"""Resident-set reporting, so the RAM claims in the README are measurable."""

from __future__ import annotations

import os
import resource
from typing import Dict


def _status() -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(("VmRSS:", "RssAnon:", "RssFile:", "RssShmem:")):
                    key, val = line.split(":", 1)
                    out[key] = int(val.strip().split()[0]) * 1024
    except OSError:
        pass
    return out


def rss_bytes() -> int:
    st = _status()
    if "VmRSS" in st:
        return st["VmRSS"]
    try:
        with open("/proc/self/statm", "rb") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def anon_rss_bytes() -> int:
    """Private, non-reclaimable memory — the part that really has to fit.

    The rest of RSS is the memory-mapped ``.safetensors``: clean, file-backed pages the
    kernel can drop and re-read at will.
    """
    return _status().get("RssAnon", rss_bytes())


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def snapshot() -> Dict[str, float]:
    return {"rss_mb": rss_bytes() / 1e6, "anon_mb": anon_rss_bytes() / 1e6,
            "peak_rss_mb": peak_rss_bytes() / 1e6}


class Tracker:
    """Records RSS at named checkpoints during a run."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.marks: list = []

    def mark(self, label: str) -> None:
        if self.enabled:
            self.marks.append((label, rss_bytes(), anon_rss_bytes(), peak_rss_bytes()))

    def report(self) -> str:
        if not self.marks:
            return ""
        w = max(len(m[0]) for m in self.marks)
        lines = [f"{'stage'.ljust(w)}   rss MB   anon MB   peak MB",
                 f"{'-' * w}   ------   -------   -------"]
        for label, rss, anon, peak in self.marks:
            lines.append(f"{label.ljust(w)}   {rss / 1e6:6.1f}   {anon / 1e6:7.1f}   "
                         f"{peak / 1e6:7.1f}")
        return "\n".join(lines)
