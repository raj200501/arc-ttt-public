#!/usr/bin/env python3
"""Re-derive every Addendum Z score from the stored model outputs, stdlib only.

Addendum Z (docs/research/ADDENDUM_Z_PROTOCOL.md) runs the +46.5 gate
again and keeps every model output. This is the script a stranger runs on
the six landed arms. It trusts no stored score. It needs the arms, this
repository's generator and scorer, and Python. Nothing is installed.

For each of the six arms present:

1. gold: regenerated from the seed with `make_task(seed, 30, 60,
   task_id="novel-0.5b-k30-seed{seed}")`, the runner's exact call. The
   stored tenant and schema text must be the regenerated ones.
2. score: every stored `prediction` is re-scored with
   `score_text_output`. `micro_f1_raw`, `valid_json` and `exact_match`
   must match exactly (the raw score is stored unrounded).
3. vote: the pooled candidates and the selected text are recomputed from
   the stored completions and per-distinct-completion log-probabilities.
   The selection must equal `prediction`.
4. mean: raw micro-F1 summed in document-index order over the scored
   documents, divided by their count, rounded to four places. The
   per-document `results` rows (what the gate reads) and every count are
   re-derived too.
5. exclusions: each carries a reason; an over-cap one a prompt longer
   than the 8192-token cap; a scored one a prompt within it.

Across the arms:

6. pairs: both arms of a seed saw the same prompt for every document
   (sha256 and length) and excluded the same over-cap documents. The gate
   is recomputed with the unchanged reader's `analyse_k` and `verdict`
   (scripts/novel_schema_summary.py) and compared with
   novel_schema_summary_2026-09-23.json when it exists.
7. readings: when novel_schema_rerun_2026-09-23.json exists, Z1 to Z4,
   the per-arm table, the predictions block and the headline are
   re-derived from the arms and the banked 2026-08-12 run.
8. run log: when addendum_z_run_log.json is present, every logged prefix
   hash and adapter digest is checked against the landed arms (its own
   `verify`), and the code this script imports must hash to the
   protocol's freeze commit as the log recorded it.

What it cannot check: that the completions and their log-probabilities
came from the model. That needs the weights and a re-run, and sampling is
unseeded (spec B.9.6), so a re-run draws a different pool. The run log
shows that documents logged during the run were not redrawn afterwards.

Blind by default. The protocol forbids quoting an arm-level number before
the reading lands. Until novel_schema_rerun_2026-09-23.json exists, no
score, mean or verdict is printed and mismatch lines carry no values.
--unblind overrides that.

    python3 scripts/verify_z_from_primary.py
    python3 scripts/verify_z_from_primary.py --dir DIR --banked-dir DIR --run-log PATH

Exit 0: all six arms present and every check matched. 1: something did
not match. 2: withheld (an arm is missing) and nothing present mismatched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Both readers keep torch inside their run functions; their module scope is
# stdlib, so importing them here costs nothing and keeps every constant
# (names, seeds, reasons, protocol string) single-sourced.
import novel_schema_rerun as rerun  # noqa: E402
import novel_schema_summary as reader  # noqa: E402

from arcttt.novel_schema import make_task  # noqa: E402
from arcttt.scoring import json_canonical, parse_json_object, score_text_output  # noqa: E402
from arcttt.text_task import TextTaskFormatError  # noqa: E402

CAP = 8192  # run_arm's TTTConfig(max_sequence_tokens=8192); over-cap rows record it as `cap`
NO_COMPLETION = "no completion"
SHOWN = 6  # problem lines printed per arm; the count is always printed
DEPENDS_ON = (  # what this script imports or mirrors, checked against the run log's freeze pins
    "src/arcttt/scoring.py",
    "src/arcttt/novel_schema.py",
    "src/arcttt/text_task.py",
    "src/arcttt/text_ttt.py",
    "scripts/novel_schema_rerun.py",
    "scripts/novel_schema_summary.py",
)
_HEX64 = re.compile(r"[0-9a-f]{64}")


# -- the vote, replayed --------------------------------------------------------
# A stdlib copy of text_ttt's _candidate_key, vote_text_candidates and
# select_text_attempts. text_ttt imports torch at module scope, so a clean
# machine cannot import it. tests/test_verify_z_from_primary.py pins this
# copy to the original: it lifts the original functions out of text_ttt.py's
# source (no torch) and runs both on the same pools.


def candidate_key(text: str) -> str:
    """Canonical-JSON pooling key; an unparseable text pools by its raw form."""

    try:
        return json_canonical(parse_json_object(text))
    except TextTaskFormatError:
        return text


def vote_candidates(texts: list[str], log_probabilities: list[float]) -> list[dict]:
    """Pool by key; count, mean lp, and the highest-lp member as the text."""

    if len(texts) != len(log_probabilities):
        raise ValueError("texts and log_probabilities must align")
    pools: dict[str, list[tuple[str, float]]] = {}
    for text, lp in zip(texts, log_probabilities):
        pools.setdefault(candidate_key(text), []).append((text, lp))
    candidates = []
    for key, members in pools.items():
        candidates.append(
            {
                "text": max(members, key=lambda member: member[1])[0],
                "key": key,
                "found_count": len(members),
                "mean_log_probability": sum(lp for _, lp in members) / len(members),
            }
        )
    return candidates


def select_top1(candidates: list[dict]) -> str | None:
    """Rank by found_count + exp(mean lp), stable, highest first; take one."""

    ranked = sorted(
        candidates,
        key=lambda c: c["found_count"] + math.exp(c["mean_log_probability"]),
        reverse=True,
    )
    return ranked[0]["text"] if ranked else None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def replay_vote(row: dict) -> tuple[list[dict] | None, str | None, list[str]]:
    """Candidates and selection from a row's stored pool, plus shape problems."""

    completions = row.get("completions")
    pairs = row.get("distinct_log_probabilities")
    if not isinstance(completions, list) or not all(isinstance(t, str) for t in completions):
        return None, None, ["completions missing or not a list of strings"]
    if not completions:
        return None, None, ["scored with an empty pool"]
    problems = []
    if len(completions) > rerun.POOL_SAMPLES:
        problems.append(f"{len(completions)} completions in a pool of at most {rerun.POOL_SAMPLES}")
    if any(not text or text != text.strip() for text in completions):
        problems.append("a completion that predict_text would have stripped or dropped")
    if not isinstance(pairs, list) or not all(
        isinstance(p, list) and len(p) == 2 and isinstance(p[0], str) and _is_number(p[1]) for p in pairs
    ):
        return None, None, problems + ["distinct_log_probabilities missing or malformed"]
    distinct = list(dict.fromkeys(completions))
    if [p[0] for p in pairs] != distinct:
        return None, None, problems + [
            "distinct_log_probabilities does not list each distinct completion once, in first-seen order"
        ]
    if any(lp > 0 for _, lp in pairs):
        problems.append("a log-probability above zero, which no model produces")
    lp_by_text = {text: lp for text, lp in pairs}
    candidates = vote_candidates(completions, [lp_by_text[text] for text in completions])
    try:
        selected = select_top1(candidates)
    except OverflowError:
        return None, None, problems + ["a log-probability too large to rank"]
    return candidates, selected, problems


# -- reporting -------------------------------------------------------------------


def _canon(value: object) -> str:
    """Exact comparison through JSON: floats by repr, NaN equal to NaN, key order ignored."""

    return json.dumps(value, sort_keys=True)


def _short(value: object, width: int = 72) -> str:
    text = repr(value)
    return text if len(text) <= width else text[: width - 3] + "..."


class Problems:
    """Collected problems. Blind: a score mismatch is named, its values are not."""

    def __init__(self, blind: bool) -> None:
        self.blind = blind
        self.lines: list[str] = []

    def add(self, line: str) -> None:
        self.lines.append(line)

    def differs(self, where: str, field: str, stored: object, derived: object, what: str | None = None) -> None:
        head = f"{where}: {what}" if what else f"{where}: stored {field} differs from the re-derived one"
        if self.blind:
            self.lines.append(f"{head} (values withheld: blind)")
        else:
            self.lines.append(f"{head}: stored {_short(stored)}, re-derived {_short(derived)}")


def _rows_by_index(record: dict) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for row in record.get("predictions") or []:
        index = row.get("index") if isinstance(row, dict) else None
        if isinstance(index, int) and 0 <= index < rerun.EVAL_N and index not in rows:
            rows[index] = row
    return rows


# -- one arm ----------------------------------------------------------------------


def verify_arm(record: dict, seed: int, arm: str, task, schema, blind: bool) -> dict:
    """Every check on one arm. Returns counts, the re-derived mean and the problems."""

    out = Problems(blind)
    identity = {
        "protocol": rerun.PROTOCOL,
        "rung": rerun.RUNG,
        "model": rerun.MODEL_ID,
        "model_revision": rerun.MODEL_REVISION,
        "arm": arm,
        "k": rerun.K,
        "eval_n": rerun.EVAL_N,
        "seed": seed,
        "device": "cpu",
        "dtype": "torch.float32",
        "replicates": rerun.artifact_name(seed, arm, rerun.BANKED_DATE),
        "tenant": schema.tenant_id,
        "schema": schema.describe(),
    }
    for field, want in identity.items():
        if record.get(field) != want:
            out.add(f"{field}: stored {_short(record.get(field))}, the protocol and seed {seed} give {_short(want)}")
    config = record.get("config") if isinstance(record.get("config"), dict) else {}
    recipe = {
        "epochs": 1 if arm == "adapted" else 0,
        "max_seq": CAP,
        "max_new_tokens": 512,
        "rank": 16,
        "alpha": 32,
        "shuffle_seeds": [seed],
    }
    for field, want in recipe.items():
        if config.get(field) != want:
            out.add(f"config.{field}: stored {_short(config.get(field))}, the frozen recipe gives {want!r}")
    adapter = record.get("adapter") if isinstance(record.get("adapter"), dict) else {}
    digest = adapter.get("sha256")
    if arm == "adapted" and not (isinstance(digest, str) and _HEX64.fullmatch(digest)):
        out.add("adapter.sha256 missing: an adapted arm records its adapter's digest")
    if arm == "kshot" and digest is not None:
        out.add("adapter.sha256 set on a k-shot arm, which trains no adapter")

    counts = {"docs": 0, "scored": 0, "over_cap": 0, "empty_pool": 0, "rescored": 0, "replayed": 0}
    rows = record.get("predictions")
    if not isinstance(rows, list):
        out.add("no `predictions` list: not a primary-verifiable Addendum Z arm")
        return {"counts": counts, "mean": None, "problems": out.lines}
    if [row.get("index") if isinstance(row, dict) else None for row in rows] != list(range(rerun.EVAL_N)):
        out.add(f"predictions are not exactly documents 0..{rerun.EVAL_N - 1}, once each, in index order")
    by_index = _rows_by_index(record)
    counts["docs"] = len(by_index)

    results: list[dict] = []
    f1_sum = 0.0
    exact = invalid = 0
    reasons = {rerun.OVER_CAP: 0, rerun.EMPTY_POOL: 0}
    for index in sorted(by_index):
        row = by_index[index]
        where = f"doc {index}"
        gold = task.test[index].output_text
        tokens = row.get("prompt_tokens")
        if not (isinstance(tokens, int) and not isinstance(tokens, bool)) or not (
            isinstance(row.get("prompt_sha256"), str) and _HEX64.fullmatch(row["prompt_sha256"])
        ):
            out.add(f"{where}: prompt_tokens or prompt_sha256 missing")
            tokens = None
        if "error" in row:
            # excluded from the mean, never scored 0 (B.9.2)
            reason = row.get("reason")
            if row["error"] != NO_COMPLETION:
                out.add(f"{where}: error {_short(row['error'])}, the runner writes {NO_COMPLETION!r}")
            if "prediction" in row:
                out.add(f"{where}: an excluded document carries a prediction")
            if reason is None:
                out.add(f"{where}: excluded without a reason")
            elif reason == rerun.OVER_CAP:
                counts["over_cap"] += 1
                reasons[reason] += 1
                if row.get("cap") != CAP:
                    out.add(f"{where}: over-cap row records cap {_short(row.get('cap'))}, the recipe's is {CAP}")
                if tokens is not None and tokens <= CAP:
                    out.add(f"{where}: excluded as over the cap, but prompt_tokens {tokens} is not above {CAP}")
                if "completions" in row:
                    out.add(f"{where}: an over-cap document was never decoded, but carries completions")
            elif reason == rerun.EMPTY_POOL:
                counts["empty_pool"] += 1
                reasons[reason] += 1
                if row.get("completions") != [] or row.get("candidates") != []:
                    out.add(f"{where}: excluded for an empty pool, but the stored pool is not empty")
                if tokens is not None and tokens > CAP:
                    out.add(f"{where}: decoded, but prompt_tokens {tokens} is above the cap {CAP}")
            else:
                out.add(f"{where}: unknown exclusion reason {_short(reason)}")
            results.append({"index": index, "error": NO_COMPLETION})
            continue
        if "prediction" not in row or not isinstance(row["prediction"], str):
            out.add(f"{where}: neither a prediction nor an exclusion")
            continue
        counts["scored"] += 1
        if tokens is not None and tokens > CAP:
            out.add(f"{where}: scored, but prompt_tokens {tokens} is above the cap {CAP}")
        prediction = row["prediction"]
        score = score_text_output(prediction, gold)
        matched = True
        stored_f1 = row.get("micro_f1_raw")
        if not _is_number(stored_f1) or stored_f1 != score.micro_f1:
            out.differs(where, "micro_f1_raw", stored_f1, score.micro_f1)
            matched = False
        for field, derived in (("valid_json", score.valid_json), ("exact_match", score.exact_match)):
            if row.get(field) is not derived:
                out.differs(where, field, row.get(field), derived)
                matched = False
        counts["rescored"] += matched
        candidates, selected, shape = replay_vote(row)
        for line in shape:
            out.add(f"{where}: {line}")
        if candidates is not None:
            same_pool = _canon(row.get("candidates")) == _canon(candidates)
            if not same_pool:
                out.add(f"{where}: stored candidates differ from those pooled from its completions and log-probabilities")
            if selected != prediction:
                out.differs(where, "prediction", prediction, selected,
                            what="the vote over its stored pool selects a different text than the stored prediction")
            counts["replayed"] += same_pool and selected == prediction and not shape
        f1_sum += score.micro_f1  # document-index order, as the runner sums
        exact += int(score.exact_match)
        invalid += int(not score.valid_json)
        results.append({"index": index, "valid_json": score.valid_json,
                        "exact_match": score.exact_match, "micro_f1": round(score.micro_f1, 4)})

    stored_results = record.get("results")
    if _canon(stored_results) != _canon(results):
        stored_list = stored_results if isinstance(stored_results, list) else []
        first = next((i for i in range(max(len(stored_list), len(results)))
                      if i >= len(stored_list) or i >= len(results)
                      or _canon(stored_list[i]) != _canon(results[i])), None)
        out.differs(f"results[{first}]", "row",
                    stored_list[first] if first is not None and first < len(stored_list) else None,
                    results[first] if first is not None and first < len(results) else None,
                    what="stored results row differs from the re-derived one (the gate reads these rows)")
    scored = counts["scored"]
    mean = round(f1_sum / scored, 4) if scored else 0.0
    if record.get("mean_micro_f1") != mean:
        out.differs("arm", "mean_micro_f1", record.get("mean_micro_f1"), mean)
    validity = "ok"
    if arm == "kshot":  # B.5, judged on the baseline arm
        validity = "floor" if mean < rerun.FLOOR else "ceiling" if mean > rerun.CEILING else "ok"
    for field, derived in (("exact_match", exact), ("invalid_json", invalid), ("validity", validity)):
        if record.get(field) != derived:
            out.differs("arm", field, record.get(field), derived)
    excluded = len(by_index) - scored
    for field, derived in (("scored", scored), ("no_completion", excluded), ("no_completion_reasons", reasons)):
        if _canon(record.get(field)) != _canon(derived):
            out.add(f"arm: stored {field} {_short(record.get(field))}, re-derived {derived!r}")
    return {"counts": counts, "mean": mean, "problems": out.lines}


# -- across the arms ----------------------------------------------------------------


def check_pair(seed: int, adapted: dict, kshot: dict) -> list[str]:
    """Both arms of a seed: one corpus, one prompt per document, one over-cap set."""

    problems = []
    for field in ("tenant", "schema", "eval_n", "device", "dtype", "model_revision"):
        if adapted.get(field) != kshot.get(field):
            problems.append(f"seed {seed}: the two arms differ in {field}")
    a, k = _rows_by_index(adapted), _rows_by_index(kshot)
    differ = [i for i in sorted(set(a) & set(k))
              if (a[i].get("prompt_sha256"), a[i].get("prompt_tokens"))
              != (k[i].get("prompt_sha256"), k[i].get("prompt_tokens"))]
    if differ:
        problems.append(f"seed {seed}: the two arms saw different prompts for {len(differ)} document(s) "
                        f"(first: doc {differ[0]}); both arms decode the same 30-demonstration prompt")
    over_a = sorted(i for i, r in a.items() if r.get("reason") == rerun.OVER_CAP)
    over_k = sorted(i for i, r in k.items() if r.get("reason") == rerun.OVER_CAP)
    if over_a != over_k:
        problems.append(f"seed {seed}: over-cap exclusions differ between the arms (adapted {over_a}, "
                        f"kshot {over_k}); the cap is a property of the corpus and the tokenizer")
    return problems


def check_gate(directory: Path, six: dict, out: Problems) -> tuple[str, str, dict]:
    """The unchanged reader's verdict on the six arms, and the summary file against it."""

    row = reader.analyse_k(six, rerun.K)
    word, detail = reader.verdict(row)
    loaded = reader.load(directory, rerun.DATE)  # what the reader's own glob picks up
    if set(loaded) != set(six) or any(_canon(loaded[key]) != _canon(six[key]) for key in six):
        out.add(f"the unchanged reader's glob over {directory.name}/novel_schema_*_{rerun.DATE}.json "
                "does not pick up exactly the six arm files")
    path = directory / f"novel_schema_summary_{rerun.DATE}.json"
    if path.exists():
        stored = json.loads(path.read_text())
        expected = {
            "date": rerun.DATE,
            "VERDICT": word,
            "verdict_detail": detail,
            "gate_k30": row,
            "comparability_k10": reader.analyse_k(loaded, 10),
        }
        for field, want in expected.items():
            if _canon(stored.get(field)) != _canon(want):
                out.differs(path.name, field, stored.get(field), want)
    return word, detail, row


def check_readings(path: Path, six: dict, word: str, row: dict, banked_dir: Path, out: Problems) -> None:
    """Z1 to Z4, the arms table, the predictions block and the headline, re-derived."""

    stored = json.loads(path.read_text())
    banked, banked_problems = rerun._six(banked_dir, rerun.BANKED_DATE)
    banked_summary_path = banked_dir / f"novel_schema_summary_{rerun.BANKED_DATE}.json"
    if not banked_summary_path.exists():
        banked_problems.append(f"missing: {banked_summary_path.name}")
    if banked_problems:
        out.add(f"{path.name}: cannot re-derive Z2 to Z4 without the banked run ({'; '.join(banked_problems)})")
        return
    banked_summary = json.loads(banked_summary_path.read_text())
    z4 = rerun.attrition_reading(six, banked)
    comparable = z4["reading"] == "SAME EXCLUSIONS"
    expected: dict = {"Z1_gate.reading": word, "Z1_gate.gate_k30": row, "Z4_attrition": z4}
    if comparable:
        try:
            z2 = rerun.agreement_reading({"VERDICT": word, "gate_k30": row}, banked_summary)
        except ValueError as error:
            out.add(f"{path.name}: Z2 cannot be re-derived ({error})")
            return
        z3 = rerun.spread_reading(six, banked)
        expected.update({"Z2_agreement": z2, "Z3_spread": z3})
    else:
        z2 = {"reading": "NOT COMPARABLE"}
        z3 = {"reading": "NOT COMPARABLE"}
        expected.update({"Z2_agreement.reading": z2["reading"], "Z3_spread.reading": z3["reading"],
                         "Z2_agreement.new_mean_delta": row.get("mean_delta"),
                         "Z2_agreement.banked_mean_delta": banked_summary["gate_k30"].get("mean_delta")})
    expected["preregistered"] = {
        "Z1": {"predicted": "GO", "observed": word, "holds": word == "GO"},
        "Z2": {"predicted": "AGREES", "observed": z2["reading"], "holds": z2["reading"] == "AGREES"},
        "Z3": {"predicted": f"max per-arm spread <= {rerun.SPREAD_PREDICTION}",
               "observed": z3.get("max_abs_diff"), "holds": bool(z3.get("prediction_holds", False))},
        "Z4": {"predicted": "SAME EXCLUSIONS", "observed": z4["reading"],
               "holds": z4["reading"] == "SAME EXCLUSIONS"},
    }
    expected["headline.banked"] = banked_summary["gate_k30"].get("mean_delta")
    expected["headline.primary_verifiable"] = row.get("mean_delta")
    expected["headline.quote"] = row.get("mean_delta") if (word == "GO" and comparable) else None
    for seed, arm in rerun.ARM_ORDER:
        record = six[(rerun.K, seed, arm)]
        for field in ("mean_micro_f1", "scored", "no_completion", "restarts", "adapt_seconds", "compute_seconds"):
            expected[f"arms.{rerun.artifact_name(seed, arm)}.{field}"] = record.get(field)
    for dotted, want in expected.items():
        value: object = stored
        for part in _split_dotted(dotted):
            value = value.get(part) if isinstance(value, dict) else None
        if _canon(value) != _canon(want):
            out.differs(path.name, dotted, value, want)


def _split_dotted(dotted: str) -> list[str]:
    """'arms.<file>.json.scored' -> ['arms', '<file>.json', 'scored'] (arm names hold dots)."""

    if dotted.startswith("arms."):
        name, field = dotted[len("arms."):].rsplit(".", 1)
        return ["arms", name, field]
    return dotted.split(".")


def check_run_log(log_path: Path, directory: Path, missing: list[str], out: Problems) -> str:
    """The run log's own verify against the landed arms, then the code pins."""

    import addendum_z_run_log as run_log

    try:
        log = json.loads(log_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        out.add(f"{log_path.name}: unreadable ({error})")
        return "unreadable"
    saved = run_log.OUT
    run_log.OUT = log_path  # its verify() reads the module-level path
    try:
        found = run_log.verify(directory)
    except (KeyError, TypeError, ValueError) as error:
        found = [f"{log_path.name}: could not be checked against the arms ({error!r})"]
    finally:
        run_log.OUT = saved
    for line in found:
        if line.endswith("arm not landed") and any(name in line for name in missing):
            continue  # a withheld arm; reported once, above
        out.add(line)
    pins = log.get("code_pins") if isinstance(log.get("code_pins"), dict) else {}
    for rel in DEPENDS_ON:
        frozen = (pins.get(rel) or {}).get("at_freeze_commit")
        if frozen is None:
            out.add(f"code pin: {rel} has no freeze-commit digest in {log_path.name}")
            continue
        now = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if now != frozen:
            out.add(f"code pin: {rel} is not the file frozen with the protocol "
                    f"(sha256 {now[:12]} here, {frozen[:12]} at the freeze commit)")
    logged = sum(1 for snap in log.get("snapshots", []) for state in snap.get("arms", [])
                 if state.get("prefix_documents"))
    return (f"{len(log.get('snapshots', []))} snapshots, {logged} logged prefixes; "
            f"{len(DEPENDS_ON)} code files checked against the freeze pins")


# -- main ---------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=str(ROOT / "experiments"), help="where the six Addendum Z arms are")
    parser.add_argument("--banked-dir", default=str(ROOT / "experiments"),
                        help="the 2026-08-12 arms and summary, for re-deriving Z2 to Z4")
    parser.add_argument("--run-log", default=None,
                        help="addendum_z_run_log.json (default: beside the arms, when present)")
    parser.add_argument("--no-run-log", action="store_true", help="skip the run-log check")
    parser.add_argument("--unblind", action="store_true",
                        help="print scores and the verdict even though the reading has not landed")
    args = parser.parse_args(argv)
    directory = Path(args.dir)
    reading_path = directory / f"novel_schema_rerun_{rerun.DATE}.json"
    blind = not (args.unblind or reading_path.exists())

    six: dict = {}
    per_arm: dict = {}
    missing: list[str] = []
    tasks: dict = {}
    for seed, arm in rerun.ARM_ORDER:
        name = rerun.artifact_name(seed, arm)
        path = directory / name
        if not path.exists():
            missing.append(name)
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            per_arm[(seed, arm)] = {"counts": None, "mean": None, "problems": [f"unreadable: {error}"]}
            continue
        if not isinstance(record, dict):
            per_arm[(seed, arm)] = {"counts": None, "mean": None, "problems": ["not a JSON object"]}
            continue
        if seed not in tasks:  # the runner's exact call; both arms of a seed share it
            tasks[seed] = make_task(seed=seed, n_train=rerun.K, n_test=rerun.EVAL_N,
                                    task_id=f"novel-{rerun.RUNG}-k{rerun.K}-seed{seed}")
        task, schema = tasks[seed]
        per_arm[(seed, arm)] = verify_arm(record, seed, arm, task, schema, blind)
        six[(rerun.K, seed, arm)] = record

    shared = Problems(blind)
    pairs_checked = []
    for seed in rerun.SEEDS:
        adapted, kshot = six.get((rerun.K, seed, "adapted")), six.get((rerun.K, seed, "kshot"))
        if adapted is not None and kshot is not None:
            for line in check_pair(seed, adapted, kshot):
                shared.add(line)
            pairs_checked.append(seed)
    complete = not missing and len(six) == len(rerun.ARM_ORDER)
    gate_line = "not computed: the gate needs all six arms"
    summary_line = "not checked"
    readings_line = "not checked"
    if complete:
        try:
            word, detail, row = check_gate(directory, six, shared)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            shared.add(f"the unchanged reader could not read these arms ({error!r})")
            gate_line = "not computed: the unchanged reader could not read the arms"
            complete = False
    if complete:
        gate_line = ("recomputed with the unchanged reader; verdict withheld (blind)" if blind
                     else f"{word}. {detail}")
        summary_path = directory / f"novel_schema_summary_{rerun.DATE}.json"
        summary_line = (f"{summary_path.name} compared field by field" if summary_path.exists()
                        else f"{summary_path.name} absent: nothing to compare")
        if reading_path.exists():
            try:
                check_readings(reading_path, six, word, row, Path(args.banked_dir), shared)
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                shared.add(f"{reading_path.name}: could not be re-derived ({error!r})")
            readings_line = f"{reading_path.name}: Z1 to Z4, arms table, predictions and headline re-derived"
        else:
            readings_line = f"{reading_path.name} absent: the reading has not landed"
    elif reading_path.exists() and missing:
        shared.add(f"{reading_path.name} exists but arm(s) are missing: {', '.join(missing)}")

    log_line = "skipped (--no-run-log)"
    if not args.no_run_log:
        log_path = Path(args.run_log) if args.run_log else directory / "addendum_z_run_log.json"
        if log_path.exists():
            log_line = check_run_log(log_path, directory, missing, shared)
        elif args.run_log:
            shared.add(f"run log not found: {log_path}")
            log_line = "not found"
        else:
            log_line = f"not checked: no {log_path.name} beside the arms"

    # -- report
    print("Addendum Z, every score re-derived from the stored outputs (standard library only)")
    print(f"  arms: {directory}")
    if blind:
        print(f"  mode: BLIND. The reading has not landed ({reading_path.name} absent), so no score,")
        print("        mean or verdict is printed and mismatch lines carry no values. --unblind overrides.")
    print()
    header = f"  {'arm':<14}{'docs':>5}{'scored':>8}{'over-cap':>10}{'empty':>7}{'re-scored':>11}{'vote':>9}{'mean':>9}  result"
    print(header)
    for seed, arm in rerun.ARM_ORDER:
        label = f"seed{seed} {arm}"
        result = per_arm.get((seed, arm))
        if result is None:
            print(f"  {label:<14}{'-':>5}{'-':>8}{'-':>10}{'-':>7}{'-':>11}{'-':>9}{'-':>9}  MISSING")
            continue
        c = result["counts"]
        status = "OK" if not result["problems"] else f"{len(result['problems'])} PROBLEM(S)"
        if c is None:
            print(f"  {label:<14}{'-':>5}{'-':>8}{'-':>10}{'-':>7}{'-':>11}{'-':>9}{'-':>9}  {status}")
            continue
        mean = "blind" if blind else ("-" if result["mean"] is None else f"{result['mean']:.4f}")
        print(f"  {label:<14}{c['docs']:>5}{c['scored']:>8}{c['over_cap']:>10}{c['empty_pool']:>7}"
              f"{c['rescored']:>7}/{c['scored']:<3}{c['replayed']:>5}/{c['scored']:<3}{mean:>9}  {status}")
    for seed, arm in rerun.ARM_ORDER:
        result = per_arm.get((seed, arm))
        if result and result["problems"]:
            lines = result["problems"]
            print(f"\n  seed{seed} {arm} ({rerun.artifact_name(seed, arm)}): {len(lines)} problem(s)")
            for line in lines[:SHOWN]:
                print(f"    {line}")
            if len(lines) > SHOWN:
                print(f"    ... and {len(lines) - SHOWN} more")
    print()
    print(f"  pairs:    {'checked for seed(s) ' + ', '.join(map(str, pairs_checked)) if pairs_checked else 'no complete pair'}"
          " (same prompt per document, same over-cap documents)")
    print(f"  gate:     {gate_line}")
    print(f"  summary:  {summary_line}")
    print(f"  readings: {readings_line}")
    print(f"  run log:  {log_line}")
    if shared.lines:
        print(f"\n  across the arms: {len(shared.lines)} problem(s)")
        for line in shared.lines:
            print(f"    {line}")

    total = sum(len(r["problems"]) for r in per_arm.values()) + len(shared.lines)
    print()
    if total:
        print(f"MISMATCH: {total} problem(s). The stored arms do not re-derive as published.")
        return 1
    if missing:
        print(f"WITHHELD: {len(six)} of {len(rerun.ARM_ORDER)} Addendum Z arms present; missing "
              f"{', '.join(missing)}. The present arms were checked document by document and matched;")
        print("the gate, the summary and the readings need all six and were not computed.")
        return 2
    print("VERIFIED: every stored score, selection, mean and count re-derives from the stored outputs,")
    print("the regenerated gold and the unchanged reader.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
