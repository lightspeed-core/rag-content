#!/usr/bin/env python3
"""Append missing PyPI wheel hashes for Konflux RHOAI wheel requirements.

uv --generate-hashes against the RHOAI index can list fewer hashes than pip uses
when the build installs a Linux cp312 manylinux wheel (x86_64 or aarch64) from a mirror.

We only patch packages in EXTRA_HASH_PACKAGES so this stays a tiny diff (add
names here if CI reports a hash mismatch for another wheel).
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

EXTRA_HASH_PACKAGES = frozenset({"markupsafe"})

HASH_LINE = re.compile(r"^(\s+)--hash=sha256:([a-f0-9]+)\s*(\\)?\s*$")


def parse_pkg_header(line: str) -> tuple[str, str] | None:
    """Parse a requirement line into a distribution name and pinned version.

    Args:
        line: First line of a block (a ``name==version`` pin, optionally continued).

    Returns:
        ``(name, version)`` if the line is a ``name==version`` pin, else ``None``.
    """
    s = line.rstrip("\n").rstrip().rstrip("\\").strip()
    if "==" not in s or s.startswith("#") or s.startswith("--"):
        return None
    name, _, ver = s.partition("==")
    name, ver = name.strip(), ver.strip()
    if not name or not ver:
        return None
    return name, ver


def load_pypi_release(name: str, version: str) -> dict[str, Any] | None:
    """Fetch PyPI project metadata JSON for a release.

    Args:
        name: Distribution name as in the requirements file.
        version: Exact version string for that release.

    Returns:
        The parsed JSON object, or ``None`` if the release is not on PyPI (404).
    """
    norm = name.replace("_", "-").lower()
    url = f"https://pypi.org/pypi/{norm}/{version}/json"
    req = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": "augment-wheel-hashes (konflux)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
            return cast("dict[str, Any]", json.load(r))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def missing_cp312_linux_manylinux_hashes(
    urls: list[dict[str, Any]], existing: set[str]
) -> list[str]:
    """List sha256 digests for cp312 manylinux wheels on Linux x86_64 and aarch64 not yet pinned.

    Args:
        urls: The ``urls`` list from a PyPI release JSON response.
        existing: Hex digests already present in the requirements block.

    Returns:
        Sorted list of missing hex sha256 digests (one entry per distinct wheel file).
    """
    found: set[str] = set()
    for u in urls:
        fn = u.get("filename", "")
        if not fn.endswith(".whl"):
            continue
        dig = (u.get("digests") or {}).get("sha256")
        if not dig or dig in existing or dig in found:
            continue
        if "cp312" not in fn or "manylinux" not in fn:
            continue
        if "x86_64" not in fn and "aarch64" not in fn:
            continue
        if "i686" in fn:
            continue
        found.add(dig)
    return sorted(found)


def _append_hash_lines(out: list[str], hashes: list[str]) -> None:
    """Append ``--hash=sha256:...`` lines, preserving requirements.txt continuation style."""
    for i, sha in enumerate(hashes):
        if i == 0:
            out[-1] = out[-1].rstrip("\n").rstrip() + " \\\n"
        is_last = i == len(hashes) - 1
        if is_last:
            out.append(f"    --hash=sha256:{sha}\n")
        else:
            out.append(f"    --hash=sha256:{sha} \\\n")


def augment_block(lines: list[str]) -> list[str]:
    """Append missing wheel hash lines for allowlisted packages when needed.

    Args:
        lines: Lines for one requirement block (header plus indented ``--hash`` lines).

    Returns:
        The same lines, possibly with extra ``--hash=sha256:...`` lines appended.
    """
    if not lines:
        return lines
    first = lines[0]
    parsed = parse_pkg_header(first)
    if not parsed:
        return lines
    name, version = parsed
    canon = name.replace("_", "-").lower()
    if canon not in EXTRA_HASH_PACKAGES:
        return lines

    existing = set()
    for line in lines:
        m = HASH_LINE.match(line)
        if m:
            existing.add(m.group(2))

    data = load_pypi_release(name, version)
    if not data:
        return lines
    urls = cast("list[dict[str, Any]]", data.get("urls") or [])
    to_add = missing_cp312_linux_manylinux_hashes(urls, existing)
    if not to_add:
        return lines

    out = list(lines)
    _append_hash_lines(out, to_add)
    return out


def augment_file(path: Path) -> None:
    """Walk a hashed requirements file and augment each package block in place.

    Args:
        path: Path to ``requirements.hashes.wheel.txt`` (or compatible format).
    """
    raw = path.read_text()
    lines = raw.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() and not line[0].isspace() and parse_pkg_header(line):
            block = [line]
            i += 1
            while i < len(lines) and lines[i].startswith((" ", "\t")):
                block.append(lines[i])
                i += 1
            out.extend(augment_block(block))
            continue
        out.append(line)
        i += 1
    path.write_text("".join(out))


def main() -> None:
    """Run the augmenter: one argument, the hashed wheel requirements path."""
    if len(sys.argv) != 2:
        print(
            "usage: augment_wheel_hashes.py <requirements.hashes.wheel.txt>",
            file=sys.stderr,
        )
        sys.exit(2)
    augment_file(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
