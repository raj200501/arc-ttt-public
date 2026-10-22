#!/usr/bin/env python3
"""How often does the outbound reconciliation gate pass a made-up figure?

    python3 scripts/reconcile_false_pass_probe.py --baseline-rev 9ed86d1

Simulated review round 10 (2026-10-22) probed
`scripts/reconcile_outbound.py` and found it passed most random figures,
because some artifact somewhere holds the same number. This measures that
rate, so the copy that cites the gate can quote it instead of implying
the gate catches every invented figure.

For each claim shape the gate reads, it writes 200 random figures into a
one-sentence body and runs the gate's own `_check` on it. Each figure
lands in one of four outcomes:

- caught: the gate reports it unreconciled and would exit non-zero;
- reconciled: the gate says it traces to the evidence;
- uncheckable: the gate says it cannot test this figure (it exits zero,
  but it does not vouch for it);
- not extracted: the gate never saw it as a claim.

Only "caught" fails the gate. The figures are random, not checked for
truth, so a few may be true by chance. Each shape is run twice: with no
citation, and with a citation of one real artifact in the same clause.

Files named `*_2026-09-23.json` are left out of the evidence here, as in
the round-10 probe: they are Addendum Z artifacts and no number may be
computed from them before Z's reading.
"""

from __future__ import annotations

import argparse
import inspect
import json
import pathlib
import random
import subprocess
import types

REPO = pathlib.Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "reconcile_outbound.py"
CITED = "project_timeline.json"
SEED = 20261022
N = 200


def _load_gate(source: str) -> types.ModuleType:
    """Load a version of the gate as if it lived at scripts/."""
    module = types.ModuleType("reconcile_outbound_probe")
    module.__file__ = str(GATE)
    exec(compile(source, str(GATE), "exec"), module.__dict__)
    return module


def _sentences(rng: random.Random, n: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"integer_count": [], "decimal": [],
                                 "signed_delta": [], "percent_whole": [],
                                 "percent_one_decimal": [], "n_of_m": []}
    for _ in range(n):
        out["integer_count"].append(f"We ran {rng.randint(1, 300)} tests")
        out["decimal"].append(f"The score was 0.8{rng.randint(0, 999):03d}")
        sign = rng.choice("+-")
        out["signed_delta"].append(
            f"F1 moved {sign}{rng.randint(0, 60)}.{rng.randint(0, 9)} points")
        out["percent_whole"].append(f"Accuracy reached {rng.randint(1, 99)}%")
        out["percent_one_decimal"].append(
            f"Accuracy reached {rng.randint(1, 99)}.{rng.randint(0, 9)}%")
        m = rng.randint(2, 300)
        out["n_of_m"].append(f"{rng.randint(1, m)} of {m} runs passed")
    return out


def _haystack(gate: types.ModuleType) -> str:
    parts = []
    for name in gate.EVIDENCE_PAGES:
        path = REPO / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    for path in sorted((REPO / "experiments").glob("*.json")):
        if (path.name == "outbound_reconciliation.json"
                or path.name.startswith("reconcile_false_pass")
                or path.name.endswith("_2026-09-23.json")):
            continue
        parts.append(path.read_text(encoding="utf-8"))
    return gate._dash_normalise("\n".join(parts))


def measure(gate: types.ModuleType, n: int = N, seed: int = SEED) -> dict:
    haystack = _haystack(gate)
    # one entry per distinct value: same verdicts, far less work
    numbers = sorted(set(gate._haystack_numbers(haystack)))
    takes_uncheckable = "uncheckable" in inspect.signature(gate._check).parameters
    cache: dict = {}
    sentences = _sentences(random.Random(seed), n)
    result: dict[str, dict] = {}
    for shape, bodies in sentences.items():
        for cited in (False, True):
            tally = {"caught": 0, "reconciled": 0, "uncheckable": 0,
                     "not_extracted": 0}
            for body in bodies:
                text = body + (f" (`experiments/{CITED}`)." if cited else ".")
                unsure: list = []
                kwargs = {"uncheckable": unsure} if takes_uncheckable else {}
                reconciled, missing = gate._check(text, None, haystack, numbers,
                                                  cache, **kwargs)
                if missing:
                    tally["caught"] += 1
                elif unsure:
                    tally["uncheckable"] += 1
                elif reconciled:
                    tally["reconciled"] += 1
                else:
                    tally["not_extracted"] += 1
            key = f"{shape}{'_cited' if cited else ''}"
            tally["n"] = len(bodies)
            tally["gate_exits_zero_rate"] = round(
                1 - tally["caught"] / len(bodies), 3)
            tally["vouched_rate"] = round(tally["reconciled"] / len(bodies), 3)
            tally["example"] = text
            result[key] = tally
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-rev", default=None,
                        help="also measure the gate as of this git revision")
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--out", default=str(
        REPO / "experiments" / "reconcile_false_pass_2026-10-22.json"))
    args = parser.parse_args()
    record: dict = {
        "what": "How often scripts/reconcile_outbound.py passes a random "
                "fabricated figure, by claim shape.",
        "method": f"{args.n} random figures per shape, seed {SEED}, each in a "
                  "one-sentence body run through the gate's own _check. "
                  "'_cited' rows add a citation of "
                  f"`experiments/{CITED}` to the same clause. Figures are "
                  "random, not checked for truth. *_2026-09-23.json files "
                  "are excluded from the evidence (Addendum Z).",
        "outcomes": {
            "caught": "reported unreconciled; the gate exits non-zero",
            "reconciled": "the gate says the figure traces to the evidence",
            "uncheckable": "the gate cannot test it; exits zero, does not "
                           "vouch for it",
            "not_extracted": "the gate never reads it as a claim",
        },
        "gate_exits_zero_rate": "share of fabricated figures that do not "
                                "fail the gate (1 - caught/n)",
        "vouched_rate": "share the gate reports as reconciled",
    }
    if args.baseline_rev:
        source = subprocess.run(
            ["git", "-C", str(REPO), "show",
             f"{args.baseline_rev}:scripts/reconcile_outbound.py"],
            check=True, capture_output=True, text=True).stdout
        record["before"] = {"gate_rev": args.baseline_rev,
                            "shapes": measure(_load_gate(source), args.n)}
    record["after"] = {"gate_rev": "working tree",
                       "shapes": measure(_load_gate(GATE.read_text(encoding="utf-8")),
                                         args.n)}
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")
    for label in ("before", "after"):
        if label not in record:
            continue
        print(f"-- {label} ({record[label]['gate_rev']})")
        for key, t in record[label]["shapes"].items():
            print(f"  {key:22s} exits-zero {t['gate_exits_zero_rate']:.3f}  "
                  f"vouched {t['vouched_rate']:.3f}  caught {t['caught']:3d}  "
                  f"uncheckable {t['uncheckable']:3d}  "
                  f"not-extracted {t['not_extracted']:3d}")
    print(f"banked: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
