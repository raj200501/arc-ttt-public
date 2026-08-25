"""One estimator, one number — enforced, because it failed three times.

History of a single published figure, the k=30 receipt-level CI lower
bound, which has now been wrong for three different reasons:

  1. `verify_verdict.py` used the normal quantile (1.96) for the receipt
     interval while using t for the cluster interval in the same
     function. The authorized reader used t throughout. Gap: 0.03 F1 --
     inside the script's own 1e-3 cross-check tolerance, so nothing ever
     failed, and VERDICT.md's number disagreed with the command VERDICT.md
     told you to run.
  2. The repair transcribed a constant (1.980) instead of asking the
     reader for its quantile. It was closer, and still not t at df=157.
  3. The reader's own `t95()` was a lookup table starting at df=39 with a
     fallback that returned the invented value 2.09 below it. Addendum E
     (df=5, true t=2.5706) therefore published a cluster interval ~19%
     too narrow -- in the flattering direction. An outside reader found
     this one by recomputing from our artifacts (erratum P13).

Every one of the three was a copied constant. So the tests below pin the
absence of copies, not the value of any particular interval: t95() is
computed exactly, both readers call it, and VERDICT.md quotes what the
script prints.

Note on the banked artifact: its stored `ci95` field was computed by the
defective estimator, and it is NOT rewritten -- artifacts are frozen and
corrections sit beside them. So the check below recomputes the interval
from the artifact's raw per-receipt records, which is the primary
evidence and is unaffected. That is the repo's own rule ("summary not
trusted") applied to its own summary.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
EXP = REPO / "experiments"
DATE = "2026-08-12"
SEEDS = (1, 2, 3)

sys.path.insert(0, str(SCRIPTS))
from novel_schema_summary import t95  # noqa: E402


def _verify_verdict_output() -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "verify_verdict.py")],
        capture_output=True, text=True, cwd=REPO, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def _printed_receipt_ci() -> tuple[str, str]:
    printed = re.search(r"receipt-level 95% CI \[([\d.]+), ([\d.]+)\]",
                        _verify_verdict_output())
    assert printed, "verify_verdict.py no longer prints the receipt CI"
    return printed.group(1), printed.group(2)


def _paired_deltas_from_raw_receipts() -> list[float]:
    """Rebuild the paired deltas from primary records, ignoring summaries."""
    deltas: list[float] = []
    for seed in SEEDS:
        arms = {}
        for arm in ("adapted", "kshot"):
            path = EXP / f"novel_schema_0.5b_k30_seed{seed}_{arm}_{DATE}.json"
            arms[arm] = {
                row["index"]: row["micro_f1"]
                for row in json.loads(path.read_text())["results"]
                if "micro_f1" in row
            }
        for index in sorted(set(arms["adapted"]) & set(arms["kshot"])):
            deltas.append(arms["adapted"][index] - arms["kshot"][index])
    return deltas


def test_receipt_interval_matches_a_recompute_from_raw_receipts() -> None:
    deltas = _paired_deltas_from_raw_receipts()
    n = len(deltas)
    assert n == 158, f"expected 158 paired receipts, rebuilt {n}"
    mean = sum(deltas) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1))
    half = t95(n - 1) * sd / math.sqrt(n)
    for expected, printed in zip((mean - half, mean + half),
                                 _printed_receipt_ci()):
        assert round(expected * 100, 1) == float(printed), (
            f"reader disagreement: raw-receipt recompute "
            f"{expected * 100:.4f} vs printed {printed}")


def test_verdict_md_quotes_the_same_interval_the_script_prints() -> None:
    """The 'check it' column must not contradict the number beside it."""
    lo, hi = _printed_receipt_ci()
    verdict = (REPO / "VERDICT.md").read_text(encoding="utf-8")
    assert f"receipt-level [{lo}, {hi}]" in verdict, (
        f"VERDICT.md does not carry the interval the script prints "
        f"([{lo}, {hi}])")


def test_no_reader_carries_a_hardcoded_quantile() -> None:
    """Guard the cause of all three failures: a copied constant."""
    banned = ("1.96 * sd", "1.980 *", "4.303 *", "2.09")
    for name in ("verify_verdict.py", "addendum_e_summary.py"):
        source = (SCRIPTS / name).read_text()
        # comments narrate this history on purpose; only code counts
        code = "\n".join(line.split("#")[0] for line in source.splitlines())
        for constant in banned:
            assert constant not in code, (
                f"{name} has a hardcoded quantile ({constant!r}) again; "
                "both readers must call t95()")


def test_exactly_one_script_defines_t95() -> None:
    """The 2026-08-21 correction said the table was "deleted, not
    extended" and that "the repo has one estimator". That was true of the
    two readers it named and FALSE of the repo: `cord_paired_power.py`
    kept its own lookup table, with the same nearest-smaller-df fallback
    the correction was written about, for a day afterwards.

    Naming two files was the bug. This scans every script, so the next
    copy fails a test rather than waiting for an audit -- which is what
    closing a defect CLASS has to mean.
    """
    definers = sorted(path.name for path in SCRIPTS.glob("*.py")
                      if re.search(r"^def t95\b", path.read_text(), re.M))
    assert definers == ["novel_schema_summary.py"], (
        f"t95 is defined in {definers}; exactly one script may define it "
        "and every other caller must import that one")


def test_no_script_carries_a_student_t_lookup_table() -> None:
    """A table is the defect class, whatever the file or variable is
    called. Fingerprints are quantiles that only appear in such a table
    (df=1, 2, 5) -- the small df where the old fallback did its damage."""
    for path in sorted(SCRIPTS.glob("*.py")):
        code = "\n".join(line.split("#")[0]
                         for line in path.read_text().splitlines())
        for fingerprint in ("12.706", "4.303", "2.5706", "2.571"):
            assert fingerprint not in code, (
                f"{path.name} carries a t-quantile table entry "
                f"({fingerprint!r}); call t95() instead")


def test_t95_is_exact_not_a_table() -> None:
    """Published values, including the df the old table silently missed."""
    reference = {1: 12.7062, 2: 4.3027, 5: 2.5706, 10: 2.2281,
                 30: 2.0423, 60: 2.0003, 157: 1.9752}
    for df, want in reference.items():
        assert abs(t95(df) - want) < 5e-4, (
            f"t95({df}) = {t95(df)}, published value {want}")


def test_addendum_e_cluster_interval_uses_five_degrees_of_freedom() -> None:
    """The exact instance an outside reader caught: six seeds, df=5."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "addendum_e_summary.py")],
        capture_output=True, text=True, cwd=REPO, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    printed = re.search(r"cluster CI95 \(n=6 seeds\): \[([\d.]+), ([\d.]+)\]",
                        result.stdout)
    assert printed, "addendum_e_summary.py no longer prints the cluster CI"
    lo = float(printed.group(1))
    assert 0.309 < lo < 0.312, (
        f"cluster CI lower bound {lo} — at t=2.5706 (df=5) it is ~0.310; "
        f"the old table's 2.09 gave ~0.3275")

    # And the docs must quote what the script prints. The receipt CI went
    # wrong for hours precisely because one authority printed a number and
    # another published a different one, with nothing comparing them; the
    # E interval had the same hole until this line.
    hi = float(printed.group(2))
    # Two decimal places, matching the artifact's own stored precision and
    # tests/test_evidence_card.py. These two checks disagreed on rounding
    # for one commit on 2026-08-22 and each passed alone; one convention,
    # both readers, or the next drift hides in the gap between them.
    quoted = f"[+{lo * 100:.2f}, +{hi * 100:.2f}]"
    for name in ("VERDICT.md", "EVIDENCE.md"):
        text = (REPO / name).read_text(encoding="utf-8")
        assert quoted in text, (
            f"{name} does not carry the Addendum E cluster interval its own "
            f"script prints ({quoted})")
