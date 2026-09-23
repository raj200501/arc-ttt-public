#!/usr/bin/env python3
"""Addendum AA — Addendum X's decoder isolation, read on the four families
whose determinism gates did not fail.

Addendum X (2026-09-16) withheld itself: its determinism gate failed on
Phi-3 in both of its runs, and X's frozen protocol made a failed gate
terminal for the whole addendum. Its cells for the other four families
were banked unread: *"raw data for a successor addendum, not results of
this one"* (X's artifact). This is the successor's reader.

It uses X's frozen readings through X's own code (`read_families`, moved
out of X's `read()`), sets aside exactly one frozen rule — the gate's
terminality — for four families, and never reads Phi-3. What that choice
buys and costs is stated in the protocol; the reader enforces the parts
of it that arithmetic can enforce: Phi-3 counts against the combined X1
claim, X's unscoped HOLDS sentence is never emitted, a family whose prompt
identity fails is VOID, and a banked reading is never overwritten.

    PYTHONPATH=src python3 scripts/decoder_isolation_read.py            # the reading (once)
    PYTHONPATH=src python3 scripts/decoder_isolation_read.py --check    # preconditions only; reads nothing
    PYTHONPATH=src python3 scripts/decoder_isolation_read.py --out /tmp/aa.json   # a stranger's recomputation

Protocol: docs/research/ADDENDUM_AA_PROTOCOL.md. The stamp is taken from
its freeze line; the anchored bytes are `snapshots_ADDENDUM_AA_PROTOCOL_
<stamp>.md`, and the sha256 of this file and of X's module are quoted in
that snapshot, so the anchor covers the reading code too.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import cord_decoder_isolation as x  # noqa: E402

FAMILIES = ("qwen2.5-0.5b", "smollm2-1.7b", "granite-2b", "falcon3-1b")
EXCLUDED = {
    "phi3-mini": ("NOT READ: X's determinism gate failed on this family in both of its "
                  "runs. Addendum Y explained the first run's constrained-arm flake on "
                  "cord-004 (reduction order); the generate-comparator mismatches on "
                  "cord-001 and cord-002, present in both runs, remain unexplained. "
                  "Counted as a family where CONSTRAINT REMOVES did not fire; its HURTS "
                  "veto is unverified."),
}
PROTOCOL_PATH = REPO / "docs" / "research" / "ADDENDUM_AA_PROTOCOL.md"
OUT = REPO / "experiments" / "cord_decoder_isolation_read_2026-09-23.json"
READER_PATH = pathlib.Path(__file__).resolve()
X_MODULE_PATH = REPO / "scripts" / "cord_decoder_isolation.py"
OTS_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
ATTESTATIONS = {"bitcoin": bytes.fromhex("0588960d73d71901"),
                "pending": bytes.fromhex("83dfe30d2ef90c8e")}
STAMP_RE = re.compile(r"ADDENDUM_AA_PROTOCOL\.md\.(\d{4}-\d{2}-\d{2}T\d{4}Z)\.ots")

# X's frozen predictions (ADDENDUM_X_PROTOCOL.md, "Two predictions, written
# before the run"), in its words, split into what four families can check.
X_PREDICTIONS_VERBATIM = {
    "X1": ("V banked 0 constrained steps on every document of four families and on "
           "43 of Falcon3's 50, with 0 fallbacks everywhere. If that accounting is "
           "correct, then arms C and P must be byte-identical on every document with "
           "0 constrained steps [...] so X1 reads INERT on Qwen2.5-0.5B, SmolLM2, "
           "Granite and Phi-3, and on Falcon3 the arms differ on exactly the seven "
           "documents where the validator fired."),
    "X2": ("PATH DIVERGES on Phi-3 (V saw 32 of 50 bodies differ at bfloat16) and "
           "PATH REPRODUCES on SmolLM2. Qwen, Falcon3 and Granite are not predicted."),
}
X1_INERT_PREDICTED = ("qwen2.5-0.5b", "smollm2-1.7b", "granite-2b")
X2_REPRODUCES_PREDICTED = ("smollm2-1.7b",)
ACCOUNTING_WORDS = ("the decoder's own accounting is wrong, and that is the result — a "
                    "larger and worse one than the contrast this addendum was built to make")


# -- the anchor ---------------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def anchored_digest(ots_path: pathlib.Path) -> str | None:
    """The file sha256 an OpenTimestamps proof commits to, or None unless
    it is a version-1 sha256 file proof carrying at least one attestation."""
    blob = ots_path.read_bytes()
    head = len(OTS_MAGIC)
    if not blob.startswith(OTS_MAGIC) or len(blob) < head + 34:
        return None
    if blob[head] != 1 or blob[head + 1] != 0x08:
        return None
    if not any(tag in blob[head + 34:] for tag in ATTESTATIONS.values()):
        return None
    return blob[head + 2: head + 34].hex()


def attestation_kinds(ots_path: pathlib.Path) -> list[str]:
    blob = ots_path.read_bytes() if ots_path.exists() else b""
    return sorted(k for k, tag in ATTESTATIONS.items() if tag in blob)


def stamp_of(protocol_text: str) -> str | None:
    """The stamp the freeze line quotes; the single source of it."""
    match = STAMP_RE.search(protocol_text)
    return match.group(1) if match else None


def code_digests(anchored_text: str) -> dict:
    """The digests the anchored protocol quotes for the reading code, and
    the files as they are now."""
    quoted = dict(re.findall(r"`(scripts/[a-z_]+\.py)` sha256 `([0-9a-f]{64})`", anchored_text))
    now = {str(p.relative_to(REPO)): _sha(p.read_bytes()) for p in (READER_PATH, X_MODULE_PATH)}
    return {"quoted_in_anchored_protocol": quoted, "now": now,
            "matches": bool(quoted) and all(quoted.get(k) == v for k, v in now.items())}


def anchor_record() -> dict:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    stamp = stamp_of(text)
    record = {"stamp": stamp, "protocol_sha256_at_read": _sha(PROTOCOL_PATH.read_bytes()),
              "ok": False}
    if stamp is None:
        record["why"] = "the freeze line quotes no stamp"
        return record
    proof = PROTOCOL_PATH.with_name(f"{PROTOCOL_PATH.name}.{stamp}.ots")
    snapshot = PROTOCOL_PATH.with_name(f"snapshots_ADDENDUM_AA_PROTOCOL_{stamp}.md")
    record.update({"proof": str(proof.relative_to(REPO)), "snapshot": str(snapshot.relative_to(REPO)),
                   "attestations": attestation_kinds(proof)})
    if not proof.exists() or not snapshot.exists():
        record["why"] = "proof or anchored snapshot missing"
        return record
    anchored = anchored_digest(proof)
    snap = snapshot.read_bytes()
    record.update({
        "anchored_sha256": anchored,
        "snapshot_sha256": _sha(snap),
        "snapshot_is_what_the_proof_commits_to": anchored == _sha(snap),
        "protocol_begins_with_the_snapshot": PROTOCOL_PATH.read_bytes().startswith(snap),
    })
    code = code_digests(snap.decode("utf-8"))
    record["code"] = code
    # A calendar promise is not a timestamp. The banked reading waits for a
    # Bitcoin attestation, so it cannot precede the block that dates the
    # protocol: an attestation cannot exist before its block does.
    record["bitcoin_attested"] = "bitcoin" in record["attestations"]
    record["ok"] = bool(record["snapshot_is_what_the_proof_commits_to"]
                        and record["protocol_begins_with_the_snapshot"]
                        and code["matches"] and record["bitcoin_attested"])
    if not record["ok"]:
        record["why"] = ("the anchored snapshot, the protocol or the code digests disagree, "
                         "or the proof is not yet Bitcoin-attested (run ots upgrade)")
    return record


# -- preconditions --------------------------------------------------------------


def gate_status(family: str, scope: dict) -> dict:
    gate = json.loads((x.OUT_DIR / f"{family}_determinism.json").read_text())
    reused = scope[family]["arm_C_reused_from_V"]
    if not reused and gate["n"] == 0 and gate["passed"]:
        status = "NOT APPLICABLE"
    elif reused and gate["n"] == x.DETERMINISM_DOCS and gate["mismatches"] == 0 and gate["passed"]:
        status = "PASSED"
    elif not gate["passed"]:
        status = "FAILED"
    else:
        status = "INCONSISTENT"
    out = {"status": status, "n": gate["n"], "mismatches": gate["mismatches"]}
    if gate.get("not_applicable_because"):
        out["not_applicable_because"] = gate["not_applicable_because"]
    return out


def preconditions(scope: dict, out: pathlib.Path, banked_reading: bool) -> list[str]:
    """Every reason the reading must not run; empty means it may."""
    problems = []
    if banked_reading:
        anchor = anchor_record()
        if not anchor["ok"]:
            problems.append(f"anchor: {anchor.get('why')} "
                            f"(stamp {anchor.get('stamp')}, snapshot matches proof: "
                            f"{anchor.get('snapshot_is_what_the_proof_commits_to')}, protocol "
                            f"begins with snapshot: {anchor.get('protocol_begins_with_the_snapshot')})")
        if out.exists():
            problems.append(f"{out.name} is already banked; a reading is made once. Recompute "
                            "into another path with --out to compare.")
    elif not OUT.exists():
        # a recomputation comes AFTER the banked reading, never instead of
        # it: otherwise --out would be a way to look at the cells before
        # the protocol is anchored
        problems.append("the banked reading does not exist yet; a recomputation with --out "
                        "is only possible after it")
    for family in FAMILIES:
        gate_path = x.OUT_DIR / f"{family}_determinism.json"
        if not gate_path.exists():
            problems.append(f"missing gate cell: {gate_path.name}")
            continue
        status = gate_status(family, scope)["status"]
        if status not in ("PASSED", "NOT APPLICABLE"):
            problems.append(f"{family}: X's determinism gate is {status}")
        needed = [x.OUT_DIR / f"{family}_plain.json", x._arm_c_path(family, scope), x.CELLS[family][2]]
        if scope[family]["arm_G_runs"]:
            needed.append(x.OUT_DIR / f"{family}_generate_neutral.json")
        problems += [f"missing cell: {p.name}" for p in needed if not p.exists()]
    for path in (x.V_ARTIFACT, x.cft.SPLIT_DIR / "gold.jsonl", x.cft.SPLIT_DIR / "train.jsonl"):
        if not path.exists():
            problems.append(f"missing: {path}")
    return problems


# -- readings layered on X's -----------------------------------------------------


def prompt_identity(family: str, scope: dict, mismatches: list[str]) -> dict:
    """Per comparison: verified, not verifiable (no digest banked), or MISMATCHED."""
    const = json.loads(x._arm_c_path(family, scope).read_text())
    comparisons = {
        "C vs P": ("verified" if "per_document_prompt_sha256_16" in const
                   else "not verifiable (the reused arm C cell banks no prompt digest)"),
    }
    if scope[family]["arm_G_runs"]:
        comparisons["P vs G"] = "verified"
    else:
        comparisons["P vs G0"] = "not verifiable (the G0 comparator cell banks text only)"
    if scope[family]["modifiers"]:
        comparisons["G vs G0 (X3)"] = "not verifiable (the G0 comparator cell banks text only)"
    for label in list(comparisons):
        short = label.split(" (")[0]
        if any(m.startswith(f"{family}/") and f"({short})" in m for m in mismatches):
            comparisons[label] = "MISMATCHED"
    return comparisons


def scoped_x1_finding(x1_per_family: dict) -> str:
    """X's combine rule with Phi-3 counted against it, and X's unscoped
    HOLDS sentence made unreachable: leaving Phi-3 unread also takes it out
    of the HURTS veto, so 'no family is an exception' cannot be said."""
    combined = dict(x1_per_family)
    combined.update(EXCLUDED)
    finding = x.x1_combine(combined)
    finding = finding.replace("families tested", "families (4 read; phi3-mini not read)")
    if finding.startswith("X1 HOLDS"):
        n = sum(1 for r in x1_per_family.values() if r.startswith("CONSTRAINT REMOVES"))
        return (f"X1 HOLDS ON THE FAMILIES READ ONLY: CONSTRAINT REMOVES in {n} of 5 "
                "families (4 read; phi3-mini not read, so the condition that no family "
                "fires HURTS is unverified on 1 of 5). X's unscoped sentence -- 'removes "
                "invalid JSON ... and no family is an exception' -- is NOT licensed. "
                f"Per-family: {json.dumps(combined)}")
    return finding


def prediction_checks(rows: list[dict]) -> dict:
    by_family = {r["family"]: r for r in rows}
    return {
        "verbatim": X_PREDICTIONS_VERBATIM,
        "x1_accounting_per_document": {
            f: {"holds": r["accounting_consistent"],
                "documents_with_zero_steps_but_differing_arms": r["documents_with_zero_steps_but_differing_arms"],
                "documents_with_steps_but_identical_arms": r["documents_with_steps_but_identical_arms"]}
            for f, r in by_family.items()},
        "x1_inert_where_predicted": {f: by_family[f]["x1_reading"] for f in X1_INERT_PREDICTED},
        "x1_inert_holds": {f: by_family[f]["x1_reading"].startswith("CONSTRAINT INERT")
                           for f in X1_INERT_PREDICTED},
        "x1_falcon3_differs_on_exactly_the_documents_with_steps": {
            "differing_documents": len(by_family["falcon3-1b"]["x1_differing_documents"]),
            "holds": by_family["falcon3-1b"]["accounting_consistent"]},
        "x2_where_predicted": {f: {"reading": by_family[f]["x2_reading"],
                                   "basis": by_family[f]["x2_basis"],
                                   "holds": by_family[f]["x2_reading"].startswith("PATH REPRODUCES")}
                               for f in X2_REPRODUCES_PREDICTED},
        "not_checkable": [
            "X1 INERT on Phi-3 (not read): 3 of the 4 families X predicted INERT are checkable",
            "X1 per-document accounting on Phi-3 (not read)",
            "X2 PATH DIVERGES on Phi-3 (not read)",
        ],
    }


def licensed_sentences(rows: list[dict], x1_finding: str, x3: dict | None,
                       identity: dict, scope: dict) -> list[str]:
    hurts = [f"{r['family']}: {r['x1_reading']}" for r in rows
             if r["x1_reading"].startswith("CONSTRAINT HURTS")]
    out = ["X1 (combined; denominator 5, phi3-mini not read and counted against): "
           + x1_finding.split(" Per-family")[0]
           + (f" HURTS, named at full size: {'; '.join(hurts)}." if hurts else "")]
    for r in rows:
        family = r["family"]
        if any(v == "MISMATCHED" for v in identity[family].values()):
            out.append(f"{family}: VOID (prompt identity failed on a compared pair); no "
                       "sentence about this family's decoder isolation is licensed.")
            continue
        caveat = ""
        if scope[family]["arm_C_reused_from_V"] and r["x1_differing_documents"]:
            caveat = (" Where arms C and P differ, arm C is V's banked cell: the difference is "
                      "the constraint, or V's cell not reproducing across processes on that "
                      "document, and this reading cannot separate the two.")
        out.append(f"{family} (X1): {r['x1_reading']}.{caveat}")
        if r["x2_reading"] == "PATH REPRODUCES" and r["x2_basis"] == "token ids":
            out.append(f"{family} (X2): our loop reproduces generate token-for-token on these "
                       f"{r['n']} documents, when neither is constrained and neither has "
                       "generation defaults in play.")
        elif r["x2_reading"].startswith("PATH REPRODUCES"):
            out.append(f"{family} (X2): our loop reproduces generate's decoded text on these "
                       f"{r['n']} documents (WEAKER TEST: decoded text, not token ids).")
        else:
            note = ("" if r["x2_basis"] == "token ids" else
                    " The comparator cell carries no environment record, so path divergence "
                    "and drift in the banked cell are not separated.")
            out.append(f"{family} (X2): {r['x2_reading']} on the {r['x2_basis']} basis, on "
                       f"these {r['n']} documents; invalid outputs {r['x2_invalid_plain']} (our "
                       f"loop) against {r['x2_invalid_other_arm']} ({r['x2_compared_against']}).{note}")
    if x3 is not None:
        out.append(f"{x3['family']} (X3): {x3['reading']}")
    return out


def obligations(rows: list[dict], x3: dict | None) -> list[dict]:
    """What the protocol says must change the same day, derived from the readings."""
    owed = [{"file": "tools/README.md",
             "what": "the jsongreedy.py section carries this addendum's reading beside X's withholding"}]
    hurts = [r["family"] for r in rows if r["x1_reading"].startswith("CONSTRAINT HURTS")]
    if hurts:
        owed.append({"file": "tools/README.md",
                     "what": ("the paragraph beginning 'What it does and does not do is measured' "
                              f"names {', '.join(hurts)} as CONSTRAINT HURTS at full size")})
    if x3 is not None and str(x3.get("reading", "")).startswith("V'S ATTRIBUTION IS TOO STRONG"):
        for target in ("docs/research/ADDENDUM_V_PROTOCOL.md (dated erratum)",
                       "VERDICT.md (Addendum V row)", "CORRECTIONS.md (dated row)"):
            owed.append({"file": target, "what": "V's attribution narrowed to E of R_V, per X3"})
    return owed


def read(out: pathlib.Path = OUT) -> int:
    banked_reading = out == OUT
    scope = x._scope()
    problems = preconditions(scope, out, banked_reading)
    if problems:
        print("WITHHELD: nothing is read and nothing is written:")
        for p in problems:
            print(f"  {p}")
        return 2
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    anchor = anchor_record()
    try:
        environment = x._environment()
    except Exception as error:  # noqa: BLE001
        environment = {"unavailable": repr(error)}
    buffered = io.StringIO()
    with contextlib.redirect_stdout(buffered):
        rows, x1_per_family, x2_per_family, x3, prompt_mismatches = x.read_families(FAMILIES, scope)
    if [r["family"] for r in rows] != [f for f in x.CELLS if f in FAMILIES]:
        raise SystemExit(f"read_families returned {[r['family'] for r in rows]}, not {FAMILIES}")
    identity = {f: prompt_identity(f, scope, prompt_mismatches) for f in FAMILIES}
    void = sorted(f for f, c in identity.items() if any(v == "MISMATCHED" for v in c.values()))
    # a VOID family stays in the count as a family where REMOVES did not
    # fire: dropping it would shrink the denominator, the flattering way
    x1_finding = scoped_x1_finding({f: ("VOID (prompt identity failed)" if f in void else r)
                                    for f, r in x1_per_family.items()})
    inconsistent = sorted(r["family"] for r in rows if not r["accounting_consistent"])
    if x3 is not None and not x3.get("v_regression_count_reproduces", True):
        for key in ("regressions_explained_by_the_defaults", "residual_documents",
                    "residual_attributed_by_arithmetic"):
            x3[key] = None  # withheld: the disagreement is the finding
    first = []
    if void:
        first.append("PROMPT IDENTITY FAILED on a compared pair in " + ", ".join(void) +
                     ": those families are VOID and no sentence about them is licensed.")
    if inconsistent:
        first.append("In " + ", ".join(inconsistent) + ": " + ACCOUNTING_WORDS + ".")
    if x3 is not None and not x3.get("v_regression_count_reproduces", True):
        first.append("V's published regression count does not reproduce from its own "
                     "banked cells; X3's attribution is withheld and this publishes first.")
    head = ""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    record = {
        "what": ("Addendum AA: Addendum X's frozen readings, through X's own code, on the "
                 "four families whose determinism gates did not fail. One frozen rule set "
                 "aside: the gate's terminality."),
        "protocol": "docs/research/ADDENDUM_AA_PROTOCOL.md",
        "reads": "docs/research/ADDENDUM_X_PROTOCOL.md (readings committed 2026-09-16T05:52Z, 0e5a11c)",
        "banked_reading": banked_reading,
        "read_started_utc": started,
        "git_head": head,
        "anchor": anchor,
        "environment": environment,
        "families_read": list(FAMILIES),
        "families_not_read": EXCLUDED,
        "determinism_gate_from_x": {f: gate_status(f, scope) for f in FAMILIES},
        "published_first": first,
        "prompt_identity_per_comparison": identity,
        "void_families": void,
        "rows": rows,
        "x1_per_family": x1_per_family,
        "x1_finding": x1_finding,
        "x2_per_family": x2_per_family,
        "x2_eos_erratum": {r["family"]: {"as_first_coded": r["x2_divergent_documents_as_first_coded"],
                                         "protocol_conforming": r["x2_divergent_documents"],
                                         "documents_changed": r["x2_documents_changed_by_the_eos_erratum"]}
                           for r in rows},
        "x3": x3,
        "decoder_accounting_inconsistent_in": inconsistent,
        "x_predictions_checked": prediction_checks(rows),
        "licensed_sentences": licensed_sentences(rows, x1_finding, x3, identity, scope),
        "obligations_the_same_day": obligations(rows, x3),
        "disclosure": (
            "Addendum X's artifact records that the operator computed the per-family X1 "
            "quantities for these families by hand during X's run, for scheduling. The "
            "record names X1 only and is silent on X2 and X3, so no reading here claims to "
            "be blind. X's readings were committed before its first plain cell; X's anchor "
            "came after its run, so a stranger cannot check that order. The decision to "
            "read these cells, and every choice in this addendum, were made after the hand "
            "computation."),
        "how_to_recompute": "PYTHONPATH=src python3 scripts/decoder_isolation_read.py --out /tmp/aa.json",
        "console_of_read_families": buffered.getvalue(),
    }
    record["read_finished_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for line in first:
        print(line)
    print(buffered.getvalue(), end="")
    print(f"\n{x1_finding}")
    if x3 is not None:
        print(f"\nX3 {x3['family']}: {x3['reading']}")
    print(f"\nbanked: {out}")
    if banked_reading:
        print(f"now anchor it: ots stamp {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report the preconditions and read nothing")
    ap.add_argument("--out", default=None, help="recompute into this path instead of the banked artifact")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out) if args.out else OUT
    if args.check:
        problems = preconditions(x._scope(), out, out == OUT)
        print("\n".join(problems) if problems else "preconditions hold; the reading may run")
        return 2 if problems else 0
    return read(out)


if __name__ == "__main__":
    raise SystemExit(main())
