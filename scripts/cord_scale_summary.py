"""Roll up Addendum-A scaled-run arm artifacts into the G-E2 decision record.

Reads every cord_scale_<rung>_k<k>_seed<s>_<arm>_<date>.json in the artifact
directory, pairs adapted vs kshot within (rung, k, seed), and evaluates the
frozen decision rule: G-E2 passes at a rung iff the k=10 paired mean delta
over seeds {1,2,3} is >= +5 F1 points (0.05). Incomplete pairs are listed,
never imputed. Writes cord_scale_summary_<date>.json.

    python scripts/cord_scale_summary.py --dir experiments --date 2026-08-11
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GATE_DELTA = 0.05  # +5 F1 points, spec Addendum A / G-E2
GATE_K = 10
GATE_SEEDS = (1, 2, 3)
IDENTITY_KEYS = {"rung", "k", "seed", "arm"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    directory = Path(args.dir)

    arms: dict[tuple[str, int, int, str], dict] = {}
    non_arm: list[str] = []
    for path in sorted(directory.glob(f"cord_scale_*_{args.date}.json")):
        if path.name.startswith("cord_scale_summary"):
            continue
        record = json.loads(path.read_text())
        if not IDENTITY_KEYS.issubset(record):
            # Hand-written incident records (e.g. the 4B CPU OOM postmortem)
            # live in this directory and match the arm glob. They are not arms
            # and carry no number - list them, never parse them as one.
            non_arm.append(path.name)
            continue
        key = (record["rung"], record["k"], record["seed"], record["arm"])
        # Provenance: Kaggle-kernel arms record a "device" field; arms produced
        # by the local container driver predate it and do not. A paired delta
        # is only meaningful within one environment - dtype, library versions
        # and hardware all differ - so this tag gates pairing below.
        env = "kernel" if "device" in record else "local"
        if "error" in record:
            arms[key] = {"artifact": path.name, "error": record["error"], "env": env}
            continue
        arms[key] = {
            "artifact": path.name,
            "mean_micro_f1": record["mean_micro_f1"],
            "invalid_json": record["invalid_json"],
            "scored": record["scored"],
            "env": env,
        }

    rungs = sorted({key[0] for key in arms})
    pairs = []
    incomplete = []
    for rung in rungs:
        for k in (5, 10, 30):
            for seed in GATE_SEEDS:
                adapted = arms.get((rung, k, seed, "adapted"))
                kshot = arms.get((rung, k, seed, "kshot"))
                entry: dict[str, object] = {"rung": rung, "k": k, "seed": seed}
                if (
                    adapted is None
                    or kshot is None
                    or "error" in adapted
                    or "error" in kshot
                ):
                    entry["status"] = "incomplete"
                    entry["adapted"] = adapted
                    entry["kshot"] = kshot
                    incomplete.append(entry)
                    continue
                if adapted["env"] != kshot["env"]:
                    # HARD REFUSAL, never a warning: a cross-environment delta
                    # would silently contaminate the one number the wedge rests
                    # on (G-E2). The fix is to re-run the odd arm in the other
                    # arm's environment, not to relax this check.
                    entry["status"] = "mixed_environment"
                    entry["adapted"] = adapted
                    entry["kshot"] = kshot
                    entry["note"] = (
                        f"adapted={adapted['env']} vs kshot={kshot['env']}; "
                        "re-run one arm so the pair is homogeneous"
                    )
                    incomplete.append(entry)
                    continue
                entry["adapted"] = adapted
                entry["kshot"] = kshot
                entry["env"] = adapted["env"]
                entry["delta_micro_f1"] = round(
                    adapted["mean_micro_f1"] - kshot["mean_micro_f1"], 4
                )
                pairs.append(entry)

    decisions = {}
    for rung in rungs:
        gate_deltas = [
            pair["delta_micro_f1"]
            for pair in pairs
            if pair["rung"] == rung and pair["k"] == GATE_K
        ]
        mixed = [
            e for e in incomplete
            if e["rung"] == rung and e["k"] == GATE_K
            and e.get("status") == "mixed_environment"
        ]
        if len(gate_deltas) < len(GATE_SEEDS):
            decisions[rung] = {
                "status": f"undecidable: {len(gate_deltas)}/{len(GATE_SEEDS)} "
                "gating pairs complete",
                "blocked_by_mixed_environment": [e["seed"] for e in mixed],
            }
            continue
        mean_delta = sum(gate_deltas) / len(gate_deltas)
        decisions[rung] = {
            "k10_deltas": gate_deltas,
            "k10_mean_delta": round(mean_delta, 4),
            "g_e2_pass": mean_delta >= GATE_DELTA,
        }

    summary = {
        "spec": "ENTERPRISE_EVAL_SPEC.md Addendum A — G-E2 decision record",
        "date": args.date,
        "gate": f"mean paired delta at k={GATE_K} over seeds {list(GATE_SEEDS)} "
        f">= +{GATE_DELTA:.2f} micro-F1",
        "pairs": pairs,
        "incomplete": incomplete,
        "non_arm_files": non_arm,
        "decisions": decisions,
    }
    out = directory / f"cord_scale_summary_{args.date}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": out.name, "decisions": decisions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
