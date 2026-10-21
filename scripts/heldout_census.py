#!/usr/bin/env python3
"""Addendum AB — the held-out census of `fencecheck scan`: fetch, scan, read.

Three stages, each banked before the next, so a stranger can re-run any one:

    python3 scripts/heldout_census.py fetch --frame experiments/heldout_census_frame.json
    python3 scripts/heldout_census.py scan  --frame experiments/heldout_census_frame.json
    python3 scripts/heldout_census.py read  --frame experiments/heldout_census_frame.json

`fetch` downloads exactly the frozen version of each package (no
dependencies) and banks the sha256 of every file it got. `scan` unpacks
each distribution, runs the scanner whose sha256 the frame freezes (and
refuses if `tools/fencecheck.py` has changed since), and banks every raw
finding with its code context for adjudication. Adjudicators write one
verdict file per finding; `read` applies the frozen arithmetic: precision
= confirmed / raw findings with a Wilson 95% interval, package hit rate =
packages with at least one confirmed site / packages scanned, and the
preregistered bar words from the frame. `read` withholds until every raw
finding carries its two independent verdicts and, where they disagree, a
third.

Stdlib only. Downloads go through pip (`python3 -m pip download`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import sys
import tarfile
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SCANNER = REPO / "tools" / "fencecheck.py"
DEFAULT_WORK = REPO / "work" / "ab"
CONTEXT_LINES = 12
VERDICTS = ("CONFIRMED", "REJECTED")


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_frame(path: pathlib.Path) -> dict:
    frame = json.loads(path.read_text())
    for key in ("packages", "scanner_sha256", "prefer", "bars"):
        if key not in frame:
            raise SystemExit(f"frame lacks {key!r}")
    return frame


def _artifact_path(frame_path: pathlib.Path, stage: str) -> pathlib.Path:
    stem = frame_path.stem.replace("_frame", "")
    return frame_path.with_name(f"{stem}_{stage}.json")


# -- fetch ----------------------------------------------------------------------


def fetch(frame_path: pathlib.Path, work: pathlib.Path) -> int:
    frame = load_frame(frame_path)
    out = _artifact_path(frame_path, "fetch")
    if out.exists():
        raise SystemExit(f"{out.name} exists; fetch is made once")
    record = {"frame": frame_path.name, "packages": {}}
    for name, version in sorted(frame["packages"].items()):
        target = work / "dist" / f"{name}-{version}"
        target.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, "-m", "pip", "download", f"{name}=={version}", "--no-deps",
                "-d", str(target), "-q"]
        args += ["--only-binary", ":all:"] if frame["prefer"] == "wheel" else ["--no-binary", ":all:"]
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0 and frame["prefer"] == "wheel":
            # no wheel published: the frame's rule says fall back to the sdist, recorded
            result = subprocess.run(args[:-2] + ["--no-binary", ":all:"], capture_output=True, text=True)
        files = sorted(p for p in target.iterdir() if p.is_file())
        record["packages"][name] = {
            "version": version,
            "ok": result.returncode == 0 and bool(files),
            "files": {p.name: _sha(p) for p in files},
            "pip_stderr_tail": result.stderr[-400:] if result.returncode else "",
        }
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"banked {out.name}: {sum(v['ok'] for v in record['packages'].values())}/{len(record['packages'])} fetched")
    return 0


# -- scan -----------------------------------------------------------------------


def _unpack(archive: pathlib.Path, dest: pathlib.Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix in (".whl", ".zip"):
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if member.startswith("/") or ".." in pathlib.PurePosixPath(member).parts:
                    raise SystemExit(f"unsafe path in {archive.name}: {member}")
            z.extractall(dest)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")
    else:
        raise SystemExit(f"unknown archive type: {archive.name}")


def _context(path: pathlib.Path, line: int) -> list[str]:
    lines = path.read_text(errors="replace").splitlines()
    lo, hi = max(0, line - 1 - CONTEXT_LINES), min(len(lines), line + CONTEXT_LINES)
    return [f"{i + 1:5d}  {lines[i]}" for i in range(lo, hi)]


def scan(frame_path: pathlib.Path, work: pathlib.Path) -> int:
    frame = load_frame(frame_path)
    if _sha(SCANNER) != frame["scanner_sha256"]:
        raise SystemExit("tools/fencecheck.py is not the frozen scanner; refusing")
    fetched = json.loads(_artifact_path(frame_path, "fetch").read_text())
    out = _artifact_path(frame_path, "scan")
    if out.exists():
        raise SystemExit(f"{out.name} exists; scan is made once")
    record = {"frame": frame_path.name, "scanner_sha256": frame["scanner_sha256"], "packages": {},
              "findings": []}
    for name, info in sorted(fetched["packages"].items()):
        if not info["ok"]:
            record["packages"][name] = {"scanned": False, "why": "fetch failed"}
            continue
        src = work / "src" / f"{name}-{info['version']}"
        if src.exists():
            shutil.rmtree(src)
        dist = work / "dist" / f"{name}-{info['version']}"
        for fname, digest in info["files"].items():
            path = dist / fname
            if _sha(path) != digest:
                raise SystemExit(f"{fname} changed since fetch")
            _unpack(path, src)
        result = subprocess.run([sys.executable, str(SCANNER), "scan", "--json", str(src)],
                                capture_output=True, text=True)
        report = json.loads(result.stdout)
        found = report.get("findings", [])
        record["packages"][name] = {"scanned": True, "version": info["version"],
                                    "files_scanned": report.get("files_scanned"),
                                    "raw_findings": len(found), "exit_code": result.returncode}
        for i, f in enumerate(found):
            fpath = pathlib.Path(f["path"]) if "path" in f else pathlib.Path(f.get("file", ""))
            rel = str(fpath.relative_to(src)) if fpath.is_absolute() else str(fpath)
            line = int(f.get("line", 0))
            record["findings"].append({
                "id": f"{name}#{i:03d}", "package": name, "version": info["version"], "file": rel,
                "line": line, "scanner_says": {k: v for k, v in f.items() if k not in ("path", "file")},
                "context": _context(src / rel if not fpath.is_absolute() else fpath, line) if line else [],
            })
    record["raw_findings_total"] = len(record["findings"])
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"banked {out.name}: {record['raw_findings_total']} raw findings across "
          f"{sum(p.get('scanned', False) for p in record['packages'].values())} packages")
    return 0


# -- read -----------------------------------------------------------------------


def final_verdict(votes: list[dict]) -> str | None:
    """Two independent verdicts; a third breaks a disagreement. None = incomplete."""
    first = [v for v in votes if v.get("role") in ("A", "B")]
    if len(first) < 2:
        return None
    a, b = (v["verdict"] for v in sorted(first, key=lambda v: v["role"]))
    if a == b:
        return a
    third = [v for v in votes if v.get("role") == "C"]
    return third[0]["verdict"] if third else None


def bar_words(precision_low: float, bars: list[dict]) -> str:
    """The frozen bar whose minimum the interval's LOWER bound clears (non-flattering)."""
    chosen = None
    for bar in sorted(bars, key=lambda b: b["min_precision_lower_bound"]):
        if precision_low >= bar["min_precision_lower_bound"]:
            chosen = bar
    return chosen["words"] if chosen else "no bar cleared"


def read(frame_path: pathlib.Path) -> int:
    frame = load_frame(frame_path)
    scanned = json.loads(_artifact_path(frame_path, "scan").read_text())
    adjud_path = _artifact_path(frame_path, "adjudications")
    votes_by_id: dict[str, list[dict]] = {}
    if adjud_path.exists():
        for v in json.loads(adjud_path.read_text())["verdicts"]:
            if v["verdict"] not in VERDICTS:
                raise SystemExit(f"bad verdict {v}")
            votes_by_id.setdefault(v["id"], []).append(v)
    finals, missing = {}, []
    for f in scanned["findings"]:
        verdict = final_verdict(votes_by_id.get(f["id"], []))
        if verdict is None:
            missing.append(f["id"])
        finals[f["id"]] = verdict
    if missing:
        print(f"WITHHELD: {len(missing)} finding(s) lack a complete adjudication: {missing[:10]}")
        return 2
    n = len(finals)
    confirmed = [fid for fid, v in finals.items() if v == "CONFIRMED"]
    lo, hi = wilson(len(confirmed), n)
    pkgs = [p for p, info in scanned["packages"].items() if info.get("scanned")]
    hit = sorted({fid.split("#")[0] for fid in confirmed})
    disagreements = sum(1 for fid in finals if len({v["verdict"] for v in votes_by_id[fid]
                                                   if v["role"] in ("A", "B")}) > 1)
    record = {
        "frame": frame_path.name,
        "scanner_sha256": scanned["scanner_sha256"],
        "raw_findings": n,
        "confirmed": len(confirmed),
        "confirmed_ids": confirmed,
        "precision": round(len(confirmed) / n, 4) if n else None,
        "precision_wilson95": [round(lo, 4), round(hi, 4)],
        "packages_scanned": len(pkgs),
        "packages_with_a_confirmed_site": hit,
        "package_hit_rate": round(len(hit) / len(pkgs), 4) if pkgs else None,
        "adjudicator_disagreements": disagreements,
        "reading": bar_words(lo, frame["bars"]) if n else frame.get("zero_findings_words", "no findings"),
        "rule": "the reading uses the LOWER Wilson bound, so a small sample cannot earn a bar its "
                "interval does not clear",
    }
    out = _artifact_path(frame_path, "reading")
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("raw_findings", "confirmed", "precision",
                                              "precision_wilson95", "package_hit_rate", "reading")}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("fetch", "scan", "read"))
    ap.add_argument("--frame", required=True)
    ap.add_argument("--work", default=str(DEFAULT_WORK))
    args = ap.parse_args(argv)
    frame = pathlib.Path(args.frame)
    if args.stage == "fetch":
        return fetch(frame, pathlib.Path(args.work))
    if args.stage == "scan":
        return scan(frame, pathlib.Path(args.work))
    return read(frame)


if __name__ == "__main__":
    raise SystemExit(main())
