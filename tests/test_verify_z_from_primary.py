"""Addendum Z's stranger-facing verifier, pinned on synthetic arms.

No Z arm may be read before its reading, so every arm here is synthetic.
Each is built in the runner's exact artifact shape (the key sets are pinned
to run_arm's source below), with gold from the real make_task and with
candidates and selections from the original text_ttt vote code. That code
is lifted out of text_ttt.py's source, so torch is never imported. A
consistent set of six must verify; each tampering must fail by name.
"""

from __future__ import annotations
import __future__

import ast
import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import types
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "verify_z_from_primary.py"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from arcttt.novel_schema import make_task  # noqa: E402
from arcttt.scoring import json_canonical, parse_json_object, score_text_output  # noqa: E402
from arcttt.text_task import TextTaskFormatError  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = _load("verify_z_from_primary")
rerun = verify.rerun
reader = verify.reader

OVER_CAP_DOCS = {2: (0, 3, 7)}  # seed -> documents over the cap, in both arms
EMPTY_POOL_DOCS = {(3, "kshot"): (5,)}  # sampling attrition, one arm only
TIE_DOC = 1  # a two-two count tie in every arm, so a log-probability decides it


# -- the original vote code, lifted from text_ttt.py without importing torch ----------


def _lift_original_vote() -> dict:
    path = REPO / "src" / "arcttt" / "text_ttt.py"
    tree = ast.parse(path.read_text())
    wanted = {"TextCandidate", "_candidate_key", "vote_text_candidates", "select_text_attempts"}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted]
    assert {n.name for n in nodes} == wanted, "text_ttt.py no longer defines the vote this verifier mirrors"
    module = types.ModuleType("text_ttt_lifted")  # dataclass looks its module up in sys.modules
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update({
        "math": math,
        "dataclass": dataclass,
        "Iterable": Iterable,
        "Sequence": Sequence,
        "json_canonical": json_canonical,
        "parse_json_object": parse_json_object,
        "TextTaskFormatError": TextTaskFormatError,
    })
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, namespace)  # noqa: S102  (the repository's own source, nothing external)
    return namespace


ORIGINAL = _lift_original_vote()


def _original_vote(texts: list[str], lps: list[float]) -> tuple[list[dict], str]:
    candidates = ORIGINAL["vote_text_candidates"](texts, lps)
    selected = ORIGINAL["select_text_attempts"](candidates, attempts=1)[0]
    return [dataclasses.asdict(c) for c in candidates], selected


# -- synthetic arms in the runner's shape ------------------------------------------------


def _variants(gold: str) -> dict[str, str]:
    obj = json.loads(gold)
    groups = sorted(obj)
    wrong = json.loads(gold)
    key = min(wrong[groups[0]])
    wrong[groups[0]][key] = "Zz " + wrong[groups[0]][key]
    short = json.loads(gold)
    del short[groups[-1]][max(short[groups[-1]])]
    return {
        "gold": gold,
        "spaced": json.dumps(obj),  # the same answer in different bytes
        "reordered": json.dumps({g: obj[g] for g in reversed(groups)}, indent=1),
        "wrong": json.dumps(wrong),  # one value off
        "short": json.dumps(short),  # one field missing
        "fenced": "```json\n" + gold + "\n```",  # not JSON
        "prose": "Here is the extraction: " + gold,  # not JSON
    }


WEIGHTS = {
    "adapted": {"gold": 5, "spaced": 3, "reordered": 2, "wrong": 1, "short": 1, "fenced": 1, "prose": 0.5},
    "kshot": {"gold": 1.5, "spaced": 1, "reordered": 0.5, "wrong": 3, "short": 3, "fenced": 2, "prose": 1},
}


def _pool(rng: random.Random, gold: str, arm: str, index: int) -> tuple[list[str], dict[str, float]]:
    v = _variants(gold)
    if index == TIE_DOC:
        texts = [v["gold"], v["wrong"], v["spaced"], v["wrong"], v["fenced"]]
        return texts, {v["gold"]: -0.2, v["wrong"]: -0.6, v["spaced"]: -0.3, v["fenced"]: -1.5}
    names, weights = zip(*WEIGHTS[arm].items())
    size = rng.randint(3, 4) if rng.random() < 0.15 else rerun.POOL_SAMPLES
    texts = [v[name] for name in rng.choices(names, weights=weights, k=size)]
    return texts, {text: rng.uniform(-2.5, -0.02) for text in dict.fromkeys(texts)}


def _arm_record(seed: int, arm: str) -> dict:
    task, schema = make_task(seed=seed, n_train=30, n_test=60, task_id=f"novel-0.5b-k30-seed{seed}")
    rng = random.Random(f"{seed}-{arm}")
    journal = []
    for index in range(len(task.test)):
        gold = task.test[index].output_text
        prompt_sha = hashlib.sha256(f"prompt {seed} {index}".encode()).hexdigest()
        if index in OVER_CAP_DOCS.get(seed, ()):
            journal.append({"index": index, "error": "no completion", "reason": rerun.OVER_CAP,
                            "prompt_tokens": 8300 + index, "prompt_sha256": prompt_sha, "cap": 8192,
                            "seconds": 0.4})
            continue
        tokens = 8000 + index
        if index in EMPTY_POOL_DOCS.get((seed, arm), ()):
            journal.append({"index": index, "error": "no completion", "reason": rerun.EMPTY_POOL,
                            "prompt_tokens": tokens, "prompt_sha256": prompt_sha, "seconds": 300.0,
                            "completions": [], "distinct_log_probabilities": [], "candidates": []})
            continue
        texts, lp_of = _pool(rng, gold, arm, index)
        candidates, selected = _original_vote(texts, [lp_of[t] for t in texts])
        score = score_text_output(selected, gold)
        journal.append({
            "index": index, "prompt_tokens": tokens, "prompt_sha256": prompt_sha,
            "completions": texts,
            "distinct_log_probabilities": [[t, lp_of[t]] for t in dict.fromkeys(texts)],
            "candidates": candidates,
            "prediction": selected, "valid_json": score.valid_json, "exact_match": score.exact_match,
            "micro_f1_raw": score.micro_f1, "seconds": 400.0,
        })
    # aggregation exactly as run_arm writes it
    results = []
    exact = invalid = scored = no_completion = 0
    f1_sum = 0.0
    for row in journal:
        if "error" in row:
            no_completion += 1
            results.append({"index": row["index"], "error": row["error"]})
            continue
        scored += 1
        exact += int(row["exact_match"])
        invalid += int(not row["valid_json"])
        f1_sum += row["micro_f1_raw"]
        results.append({"index": row["index"], "valid_json": row["valid_json"],
                        "exact_match": row["exact_match"], "micro_f1": round(row["micro_f1_raw"], 4)})
    mean_f1 = round(f1_sum / scored, 4)
    validity = "ok"
    if arm == "kshot":
        validity = "floor" if mean_f1 < 0.15 else "ceiling" if mean_f1 > 0.95 else "ok"
    return {
        "spec": rerun.SPEC, "protocol": rerun.PROTOCOL,
        "dataset": "synthetic novel-schema tenants (novel_schema.py), no external data",
        "tenant": schema.tenant_id, "schema": schema.describe(), "rung": "0.5b",
        "model": rerun.MODEL_ID, "model_revision": rerun.MODEL_REVISION, "arm": arm, "k": 30,
        "eval_n": 60, "seed": seed, "gate_role": "DECISION",
        "replicates": rerun.artifact_name(seed, arm, rerun.BANKED_DATE), "validity": validity,
        "decode": "synthetic (tests/test_verify_z_from_primary.py)",
        "config": {"rank": 16, "alpha": 32, "epochs": 1 if arm == "adapted" else 0, "max_new_tokens": 512,
                   "max_seq": 8192, "chunked_loss_tokens": 512, "gradient_checkpointing": True,
                   "shuffle_examples": True, "shuffle_seeds": [seed], "learning_rate": 5e-5},
        "device": "cpu", "dtype": "torch.float32",
        "environment": {"note": "synthetic"},
        "adapter": {"training_examples": 30, "training_examples_within_cap": 30,
                    "sha256": hashlib.sha256(f"adapter {seed}".encode()).hexdigest() if arm == "adapted" else None,
                    "file": f"{rerun.ckpt_stem(seed, arm)}_adapter.pt" if arm == "adapted" else None,
                    "initialisation": "synthetic"},
        "resumed": False, "restarts": 0, "started_utc": "2026-09-23T03:00:00Z",
        "banked_utc": "2026-09-24T03:00:00Z", "adapt_seconds": 1.0 if arm == "adapted" else 0.0,
        "decode_seconds": 2.0, "compute_seconds": 3.0, "last_process_wall_seconds": 3.0,
        "exact_match": exact, "scored": scored, "invalid_json": invalid, "no_completion": no_completion,
        "no_completion_reasons": {
            rerun.OVER_CAP: sum(1 for r in journal if r.get("reason") == rerun.OVER_CAP),
            rerun.EMPTY_POOL: sum(1 for r in journal if r.get("reason") == rerun.EMPTY_POOL),
        },
        "mean_micro_f1": mean_f1, "results": results, "predictions": journal,
    }


BASE = {(seed, arm): _arm_record(seed, arm) for seed, arm in rerun.ARM_ORDER}


def _records() -> dict:
    return copy.deepcopy(BASE)


def _write(tmp_path: Path, records: dict, summary: bool = True) -> Path:
    out = tmp_path / "arms"
    out.mkdir()
    for (seed, arm), record in records.items():
        (out / rerun.artifact_name(seed, arm)).write_text(json.dumps(record, indent=2))
    if summary:  # the unchanged reader writes it, as the --read step does
        reader.main(["--dir", str(out), "--date", rerun.DATE])
    return out


def _edit(out: Path, seed: int, arm: str, change) -> None:
    path = out / rerun.artifact_name(seed, arm)
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record, indent=2))


def _verify(capsys, out: Path, *extra: str) -> tuple[int, str]:
    capsys.readouterr()  # drop the setup's own prints
    code = verify.main(["--dir", str(out), "--banked-dir", str(out.parent / "banked"), *extra])
    return code, capsys.readouterr().out


def _scored(record: dict, index: int) -> dict:
    row = record["predictions"][index]
    assert "prediction" in row
    return row


# -- the copy of the vote is the original vote -----------------------------------------------

POOL_TEXTS = [
    '{"a": 1}', '{"a":1}', '{ "a" : 1 }', '{"a": 1, "b": 2}', '{"b": 2, "a": 1}', '{"a": 2}',
    '{"a": 1, "a": 2}', "not json", "[1, 2]", '"a string"', '{"a": 1} trailing',
    '```json\n{"a": 1}\n```', '{"a": {"b": [1, 2]}}', '{"a":{"b":[1,2]}}', '{"a": NaN}', "{}",
]


def _random_pool(rng: random.Random) -> tuple[list[str], list[float]]:
    texts = [rng.choice(POOL_TEXTS) for _ in range(rng.randint(1, 5))]
    ties = [-0.1, -0.5, -1.0, -2.0, 0.0, -30.0, -1e-9]
    lp_of = {t: rng.choice(ties) if rng.random() < 0.5 else rng.uniform(-4.0, 0.0) for t in dict.fromkeys(texts)}
    return texts, [lp_of[t] for t in texts]


def test_the_vote_copy_is_the_original_vote_code_on_random_pools():
    rng = random.Random(20260923)
    for _ in range(2000):
        texts, lps = _random_pool(rng)
        original, selected = _original_vote(texts, lps)
        mine = verify.vote_candidates(texts, lps)
        assert mine == original
        assert verify.select_top1(mine) == selected
        for text in texts:
            assert verify.candidate_key(text) == ORIGINAL["_candidate_key"](text)


def test_the_vote_copy_matches_the_imported_text_ttt_where_torch_is_already_loaded():
    # Never imports torch itself: it runs when the full suite has already
    # loaded it, and skips when this file runs alone, so running this file
    # on a box mid-experiment cannot pull torch in.
    if "torch" not in sys.modules:
        pytest.skip("torch not loaded in this process; the lifted-source test above covers the pin")
    from arcttt import text_ttt

    rng = random.Random(7)
    for _ in range(500):
        texts, lps = _random_pool(rng)
        original = text_ttt.vote_text_candidates(texts, lps)
        mine = verify.vote_candidates(texts, lps)
        assert mine == [dataclasses.asdict(c) for c in original]
        assert verify.select_top1(mine) == text_ttt.select_text_attempts(original, attempts=1)[0]


# -- the synthetic arms have the runner's exact shape --------------------------------------------


def _literal_keys(function: str, variable: str, source: Path) -> list[frozenset]:
    tree = ast.parse(source.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    shapes = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets)):
            shapes.append([None if k is None else k.value for k in node.value.keys])
    return shapes


def test_the_synthetic_arms_carry_exactly_the_keys_run_arm_writes():
    runner = REPO / "scripts" / "novel_schema_rerun.py"
    (trace,) = _literal_keys("voted_with_trace", "trace", runner)
    rows = {frozenset(k for k in keys if k is not None) | (frozenset(trace) if None in keys else frozenset())
            for keys in _literal_keys("run_arm", "row", runner)}
    (report,) = _literal_keys("run_arm", "report", runner)
    record = BASE[(3, "kshot")]
    built = {frozenset(record["predictions"][i]) for i in (0, 5)}  # one scored, one empty pool
    built.add(frozenset(BASE[(2, "adapted")]["predictions"][0]))  # one over the cap
    assert built == rows
    assert set(record) == set(report)


def test_the_synthetic_gate_is_decidable_and_the_tie_document_is_a_tie():
    word, _ = reader.verdict(reader.analyse_k({(30, s, a): r for (s, a), r in BASE.items()}, 30))
    assert word == "GO"  # so the Z2 reading below is defined (the banked word is GO by construction)
    counts = [c["found_count"] for c in BASE[(1, "adapted")]["predictions"][TIE_DOC]["candidates"]]
    assert counts == [2, 2, 1]


# -- a consistent set verifies, blind by default --------------------------------------------------


def test_six_consistent_arms_verify_and_nothing_is_printed_while_blind(tmp_path, capsys):
    out = _write(tmp_path, _records())
    code, text = _verify(capsys, out)
    assert code == 0, text
    assert "VERIFIED" in text and "BLIND" in text and "verdict withheld" in text
    assert "compared field by field" in text
    for record in BASE.values():
        assert f"{record['mean_micro_f1']:.4f}" not in text
    assert "Gate PASSES" not in text and "Gate FAILS" not in text
    assert text.count("  OK") == 6


def test_unblind_prints_the_means_and_the_verdict(tmp_path, capsys):
    out = _write(tmp_path, _records())
    code, text = _verify(capsys, out, "--unblind")
    assert code == 0, text
    assert "BLIND" not in text and "Gate PASSES" in text
    assert f"{BASE[(1, 'adapted')]['mean_micro_f1']:.4f}" in text


def test_a_missing_arm_withholds_after_checking_the_others(tmp_path, capsys):
    records = _records()
    del records[(3, "kshot")]
    out = _write(tmp_path, records, summary=False)
    code, text = _verify(capsys, out)
    assert code == 2, text
    assert "WITHHELD: 5 of 6" in text and "MISSING" in text
    assert text.count("  OK") == 5
    assert "not computed: the gate needs all six arms" in text


# -- each tampering fails by name ------------------------------------------------------------------


def test_a_changed_prediction_fails_the_replay_and_the_score(tmp_path, capsys):
    out = _write(tmp_path, _records())
    wrong = _variants(make_task(seed=1, n_train=30, n_test=60)[0].test[TIE_DOC].output_text)["wrong"]
    _edit(out, 1, "adapted", lambda r: _scored(r, TIE_DOC).update(prediction=wrong))
    code, text = _verify(capsys, out)
    assert code == 1
    assert f"doc {TIE_DOC}: the vote over its stored pool selects a different text than the stored prediction" in text
    assert f"doc {TIE_DOC}: stored micro_f1_raw differs from the re-derived one (values withheld: blind)" in text
    assert wrong not in text  # blind: no text, no value


def test_a_changed_score_fails_on_that_document_only(tmp_path, capsys):
    out = _write(tmp_path, _records())
    _edit(out, 2, "kshot", lambda r: _scored(r, 10).update(micro_f1_raw=_scored(r, 10)["micro_f1_raw"] - 0.01))
    code, text = _verify(capsys, out)
    assert code == 1
    assert "doc 10: stored micro_f1_raw differs from the re-derived one" in text
    assert f"seed2 kshot ({rerun.artifact_name(2, 'kshot')}): 1 problem(s)" in text


def test_a_changed_log_probability_that_flips_the_selection_fails(tmp_path, capsys):
    out = _write(tmp_path, _records())

    def flip(record):
        row = _scored(record, TIE_DOC)
        wrong_text = row["completions"][1]
        row["distinct_log_probabilities"] = [[t, -0.01 if t == wrong_text else -3.0]
                                             for t, _ in row["distinct_log_probabilities"]]

    _edit(out, 3, "adapted", flip)
    code, text = _verify(capsys, out)
    assert code == 1
    assert f"doc {TIE_DOC}: stored candidates differ from those pooled" in text
    assert f"doc {TIE_DOC}: the vote over its stored pool selects a different text" in text


def test_a_wrong_mean_fails(tmp_path, capsys):
    out = _write(tmp_path, _records())
    _edit(out, 1, "kshot", lambda r: r.update(mean_micro_f1=round(r["mean_micro_f1"] + 0.0001, 4)))
    code, text = _verify(capsys, out)
    assert code == 1
    assert "arm: stored mean_micro_f1 differs from the re-derived one" in text


def test_a_changed_results_row_fails_even_when_the_journal_is_intact(tmp_path, capsys):
    out = _write(tmp_path, _records())
    _edit(out, 1, "adapted", lambda r: r["results"][20].update(micro_f1=0.0))
    code, text = _verify(capsys, out)
    assert code == 1
    assert "results[20]: stored results row differs from the re-derived one (the gate reads these rows)" in text


def test_an_exclusion_without_a_reason_fails(tmp_path, capsys):
    out = _write(tmp_path, _records())
    _edit(out, 2, "adapted", lambda r: r["predictions"][3].pop("reason"))
    code, text = _verify(capsys, out)
    assert code == 1
    assert "doc 3: excluded without a reason" in text


def test_an_over_cap_exclusion_within_the_cap_fails(tmp_path, capsys):
    out = _write(tmp_path, _records())
    _edit(out, 2, "kshot", lambda r: r["predictions"][7].update(prompt_tokens=8192))
    code, text = _verify(capsys, out)
    assert code == 1
    assert "doc 7: excluded as over the cap, but prompt_tokens 8192 is not above 8192" in text
    assert "seed 2: the two arms saw different prompts for 1 document(s) (first: doc 7)" in text


def test_a_schema_that_does_not_regenerate_from_the_seed_fails(tmp_path, capsys):
    out = _write(tmp_path, _records())
    _edit(out, 3, "kshot", lambda r: r.update(schema=r["schema"].replace("->", "=>", 1)))
    code, text = _verify(capsys, out)
    assert code == 1
    assert "schema: stored" in text and "the protocol and seed 3 give" in text


def test_paired_arms_that_saw_different_prompts_fail(tmp_path, capsys):
    out = _write(tmp_path, _records())
    _edit(out, 1, "kshot", lambda r: r["predictions"][4].update(prompt_sha256="0" * 64))
    code, text = _verify(capsys, out)
    assert code == 1
    assert "seed 1: the two arms saw different prompts for 1 document(s) (first: doc 4)" in text


def test_a_tampered_summary_fails_against_the_recomputed_gate(tmp_path, capsys):
    out = _write(tmp_path, _records())
    path = out / f"novel_schema_summary_{rerun.DATE}.json"
    summary = json.loads(path.read_text())
    summary["gate_k30"]["mean_delta"] = round(summary["gate_k30"]["mean_delta"] + 0.01, 4)
    path.write_text(json.dumps(summary, indent=2))
    code, text = _verify(capsys, out)
    assert code == 1
    assert f"novel_schema_summary_{rerun.DATE}.json: stored gate_k30 differs from the re-derived one" in text


# -- the readings, written by the real reader, re-derived -------------------------------------------


def _banked(tmp_path: Path, records: dict, comparable: bool) -> Path:
    """The banked run as the new one scored it, without predictions or reasons.

    Comparable: the banked run decoded the document the new k-shot arm lost to
    an empty pool, so the over-cap sets agree (Z4 SAME EXCLUSIONS). Otherwise
    the banked arm excluded it too, and Z4 reads DIFFERENT EXCLUSIONS.
    """

    banked = tmp_path / "banked"
    banked.mkdir()
    for (seed, arm), record in records.items():
        old = {k: v for k, v in record.items() if k not in ("predictions", "protocol")}
        if comparable:
            for index in EMPTY_POOL_DOCS.get((seed, arm), ()):
                old["results"][index] = {"index": index, "valid_json": True, "exact_match": False, "micro_f1": 0.5}
        (banked / rerun.artifact_name(seed, arm, rerun.BANKED_DATE)).write_text(json.dumps(old))
    reader.main(["--dir", str(banked), "--date", rerun.BANKED_DATE])
    return banked


def _landed(tmp_path: Path, monkeypatch, comparable: bool = True) -> Path:
    records = _records()
    out = _write(tmp_path, records, summary=False)
    banked = _banked(tmp_path, records, comparable)
    monkeypatch.setattr(rerun, "environment", lambda *a, **k: {"note": "synthetic"})  # the real one imports torch
    assert rerun.read(out, banked) == 0  # the real --read writes the summary and the reading
    return out


def test_the_landed_reading_re_derives_and_unblinds(tmp_path, capsys, monkeypatch):
    out = _landed(tmp_path, monkeypatch)
    code, text = _verify(capsys, out)
    assert code == 0, text
    assert "BLIND" not in text and "Gate PASSES" in text
    assert "Z1 to Z4, arms table, predictions and headline re-derived" in text
    reading = json.loads((out / f"novel_schema_rerun_{rerun.DATE}.json").read_text())
    assert reading["Z4_attrition"]["reading"] == "SAME EXCLUSIONS"
    assert reading["Z2_agreement"]["reading"] == "AGREES"


def test_a_landed_reading_that_is_not_comparable_also_re_derives(tmp_path, capsys, monkeypatch):
    out = _landed(tmp_path, monkeypatch, comparable=False)
    reading = json.loads((out / f"novel_schema_rerun_{rerun.DATE}.json").read_text())
    assert reading["Z4_attrition"]["reading"] == "DIFFERENT EXCLUSIONS"
    code, text = _verify(capsys, out)
    assert code == 0, text
    reading["Z2_agreement"]["reading"] = "AGREES"
    (out / f"novel_schema_rerun_{rerun.DATE}.json").write_text(json.dumps(reading))
    code, text = _verify(capsys, out)
    assert code == 1
    assert "stored Z2_agreement.reading differs" in text


def test_a_tampered_reading_fails(tmp_path, capsys, monkeypatch):
    out = _landed(tmp_path, monkeypatch)
    path = out / f"novel_schema_rerun_{rerun.DATE}.json"
    reading = json.loads(path.read_text())
    reading["Z3_spread"]["max_abs_diff"] = 0.0001
    reading["headline"]["quote"] = 0.5
    path.write_text(json.dumps(reading, indent=2))
    code, text = _verify(capsys, out)
    assert code == 1
    assert f"novel_schema_rerun_{rerun.DATE}.json: stored Z3_spread differs" in text
    assert f"novel_schema_rerun_{rerun.DATE}.json: stored headline.quote differs" in text


# -- the run log ------------------------------------------------------------------------------------

run_log = importlib.import_module("addendum_z_run_log")


def _write_log(out: Path, records: dict, upto: int = 20) -> Path:
    pins = {rel: {"at_freeze_commit": hashlib.sha256((REPO / rel).read_bytes()).hexdigest()}
            for rel in run_log.PINNED if (REPO / rel).exists()}
    states = [{"arm": rerun.artifact_name(s, a), "restarts": 0,
               "adapter_sha256": records[(s, a)]["adapter"]["sha256"],
               "documents_decoded": upto, "prefix_documents": upto,
               "prefix_sha256": run_log.row_digest(records[(s, a)]["predictions"][:upto])}
              for s, a in rerun.ARM_ORDER]
    path = out / "addendum_z_run_log.json"
    path.write_text(json.dumps({"code_pins": pins, "snapshots": [{"utc": "2026-09-23T12:00:00Z", "arms": states}]}))
    return path


def test_every_file_the_verifier_depends_on_is_pinned_by_the_run_log():
    assert set(verify.DEPENDS_ON) <= set(run_log.PINNED)


def test_the_run_log_matches_and_a_redrawn_prefix_or_changed_code_fails(tmp_path, capsys):
    records = _records()
    out = _write(tmp_path, records)
    log_path = _write_log(out, records)
    code, text = _verify(capsys, out)
    assert code == 0, text
    assert "1 snapshots, 6 logged prefixes" in text

    log = json.loads(log_path.read_text())
    log["snapshots"][0]["arms"][2]["prefix_sha256"] = "0" * 64
    log["code_pins"]["src/arcttt/scoring.py"]["at_freeze_commit"] = "f" * 64
    log_path.write_text(json.dumps(log))
    code, text = _verify(capsys, out)
    assert code == 1
    assert f"{rerun.artifact_name(2, 'adapted')}: the first 20 documents differ from what was logged" in text
    assert "code pin: src/arcttt/scoring.py is not the file frozen with the protocol" in text


# -- nothing but the standard library ---------------------------------------------------------------


def test_the_verifier_runs_with_no_installed_package_and_no_torch(tmp_path):
    out = _write(tmp_path, _records())
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "torch.py").write_text('raise ImportError("torch is deliberately unavailable")\n')
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "PYTHONPATH": str(stub)}
    # -S keeps site-packages off the path: a pass means stdlib plus this repository only
    proc = subprocess.run([sys.executable, "-S", str(SCRIPT), "--dir", str(out), "--no-run-log"],
                          capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120,
                          check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VERIFIED" in proc.stdout
    assert "torch" not in proc.stderr
