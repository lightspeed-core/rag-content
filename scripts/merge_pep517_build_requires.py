#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging>=24.1"]
# ///
"""Merge pyproject.toml [build-system].requires into requirements-build.txt for hermetic prefetch.

pybuild-deps only lists build tools for third-party sdists in requirements.source.txt, not the
root package. Cachi2 still needs wheels for PEP 517 install (pip install .).

Pins are resolved from PyPI's JSON API (no nested ``uv`` subprocess — avoids deadlocks under
``uv run --script``).
"""

from __future__ import annotations

import json
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

BEGIN = "# --- PEP 517 [build-system] (from pyproject.toml; merged by konflux_requirements.sh) ---"
END = "# --- end PEP 517 [build-system] ---"

USER_AGENT = "lightspeed-rag-content-merge-pep517/1.0"


def _pin_build_requirement(req_line: str) -> str:
    """Resolve one PEP 508 line to ``name==version`` using PyPI."""
    req = Requirement(req_line)
    url = f"https://pypi.org/pypi/{req.name}/json"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"PyPI: {req.name}: HTTP {e.code}") from e
    candidates: list[Version] = []
    for vstr in data.get("releases", {}):
        try:
            v = Version(vstr)
        except InvalidVersion:
            continue
        if req.specifier.contains(v, prereleases=True):
            candidates.append(v)
    if not candidates:
        raise SystemExit(f"No PyPI release satisfies {req_line!r} for {req.name!r}")
    best = max(candidates)
    return f"{canonicalize_name(req.name)}=={best}"


def _package_names_in_build_file(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            names.add(canonicalize_name(Requirement(line).name))
        except ValueError:
            continue
    return names


def _strip_previous_merge(text: str) -> str:
    if BEGIN not in text:
        return text.rstrip() + "\n"
    before, _, rest = text.partition(BEGIN)
    if END in rest:
        _, _, after = rest.partition(END)
        merged = before.rstrip() + "\n" + after.lstrip()
    else:
        merged = before.rstrip() + "\n"
    return merged if merged.endswith("\n") else merged + "\n"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    pyproject = root / "pyproject.toml"
    build_file = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "requirements-build.txt"

    if not pyproject.is_file():
        raise SystemExit(f"Not found: {pyproject}")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    raw_requires = (data.get("build-system") or {}).get("requires") or []
    body = build_file.read_text(encoding="utf-8")
    stripped = _strip_previous_merge(body)

    if not raw_requires:
        if stripped != body:
            build_file.write_text(stripped, encoding="utf-8")
        return

    existing = _package_names_in_build_file(stripped)
    missing: list[str] = []
    for line in raw_requires:
        if not isinstance(line, str) or not line.strip():
            continue
        try:
            name = canonicalize_name(Requirement(line.strip()).name)
        except ValueError:
            continue
        if name not in existing:
            missing.append(line.strip())

    if not missing:
        if stripped != body:
            build_file.write_text(stripped, encoding="utf-8")
        return

    body = stripped

    pinned_lines = [_pin_build_requirement(line) for line in missing]

    block = "\n".join([BEGIN, *pinned_lines, END, ""])
    build_file.write_text(body.rstrip() + "\n" + block, encoding="utf-8")
    print(
        f"Merged {len(pinned_lines)} PEP 517 build-system package(s) into {build_file}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
