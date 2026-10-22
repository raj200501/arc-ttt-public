"""Addendum Z's publication text, pinned before the reading exists.

`scripts/publish_z_reading.py` holds the replacement text for every page
that quotes +46.5, one version per outcome the reader can emit, and
fills it from the reading. These tests feed it synthetic readings shaped
exactly like what `novel_schema_rerun.read()` writes: the shape is taken
from read()'s source by parsing it (read() itself is never run, and
nothing here imports torch), and the values come from the runner's and
the unchanged reader's own pure functions on synthetic arms. Then they
apply the printed text to the pages and hold the result to the frozen
citation rule, so the rule cannot be applied selectively on the day the
number is known.

The pages here are frozen excerpts (tests/fixtures/z_publish/, written by
`publish_z_reading.py --freeze-fixtures`), not the live tree, so a prose
edit elsewhere cannot break these tests. Whether the live pages still
hold every anchor is the operator's `--check-anchors`, run first on
landing day; it is deliberately not a test.
"""
import ast
import hashlib
import importlib.util
import inspect
import json
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pub = _load("publish_z_reading")
z = pub.z
reader = _load("novel_schema_summary")
BANKED_SUMMARY = REPO / "experiments" / "novel_schema_summary_2026-08-12.json"
CITATION_TEST = REPO / "tests" / "test_addendum_z_citation.py"
HEADLINE_PAGES = ("VERDICT.md", "EVIDENCE.md", "README.md")
LANDED = "2026-10-24"  # synthetic


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


# -- the synthetic reading, built the way read() builds it --------------------


def _arm(seed, arm, scores, over_cap=(), empty=(), restarts=0, banked=False):
    """An arm record as run_arm banks it (banked=True: as the 2026-08-12 arms, no predictions)."""
    results, predictions = [], []
    for i, score in enumerate(scores):
        if i in over_cap or i in empty:
            results.append({"index": i, "error": "no completion"})
            predictions.append({"index": i, "error": "no completion",
                                "reason": z.OVER_CAP if i in over_cap else z.EMPTY_POOL})
            continue
        results.append({"index": i, "valid_json": True, "exact_match": False, "micro_f1": round(score, 4)})
        predictions.append({"index": i, "prediction": "{}", "valid_json": True, "exact_match": False,
                            "micro_f1_raw": score})
    scored = [r["micro_f1"] for r in results if "micro_f1" in r]
    record = {"rung": z.RUNG, "k": z.K, "seed": seed, "arm": arm, "eval_n": z.EVAL_N, "device": "cpu",
              "dtype": "torch.float32", "mean_micro_f1": round(sum(scored) / len(scored), 4),
              "scored": len(scored), "no_completion": len(scores) - len(scored), "results": results}
    if not banked:
        record.update({"protocol": z.PROTOCOL, "predictions": predictions, "restarts": restarts,
                       "adapt_seconds": 1.0 if arm == "adapted" else 0.0, "compute_seconds": 2.0})
    return record


BANKED_SEED2_OVER_CAP = tuple(range(38, 60))  # 22, as B.9.2 records (which 22 is immaterial here)


def _case(delta, kshot_level=0.5, jitter=0.0, seed2_over_cap=BANKED_SEED2_OVER_CAP, empty=()):
    """Six re-run arms, six banked arms, and both summaries."""
    new_arms, banked_arms = {}, {}
    for seed in z.SEEDS:
        kshot = [round(kshot_level + 0.02 * ((i * 7 + seed) % 5), 4) for i in range(z.EVAL_N)]
        adapted = [round(min(1.0, k + delta + (jitter if i % 2 else -jitter)), 4) for i, k in enumerate(kshot)]
        banked_adapted = [round(min(1.0, k + 0.4648), 4) for k in kshot]
        cap_new = seed2_over_cap if seed == 2 else ()
        cap_old = BANKED_SEED2_OVER_CAP if seed == 2 else ()
        new_arms[(z.K, seed, "kshot")] = _arm(seed, "kshot", kshot, cap_new)
        new_arms[(z.K, seed, "adapted")] = _arm(seed, "adapted", adapted, cap_new, empty if seed == 1 else (), 1)
        banked_arms[(z.K, seed, "kshot")] = _arm(seed, "kshot", kshot, cap_old, banked=True)
        banked_arms[(z.K, seed, "adapted")] = _arm(seed, "adapted", banked_adapted, cap_old, banked=True)
    gate = reader.analyse_k(new_arms, z.K)
    word, detail = reader.verdict(gate)
    new_summary = {"spec": "ENTERPRISE_EVAL_SPEC.md Addendum B (frozen 2026-08-12T19:40Z)", "date": z.DATE,
                   "decided": LANDED, "VERDICT": word, "verdict_detail": detail, "gate_k30": gate,
                   "comparability_k10": reader.analyse_k(new_arms, 10), "k10_promotion_bar": "..."}
    banked_summary = json.loads(BANKED_SUMMARY.read_text())
    return new_arms, banked_arms, new_summary, banked_summary


def build_reading(new_arms, banked_arms, new_summary, banked_summary) -> dict:
    """read()'s assembly, step for step (scripts/novel_schema_rerun.py, after summary_main)."""
    word = new_summary["VERDICT"]
    z4 = z.attrition_reading(new_arms, banked_arms)
    comparable = z4["reading"] == "SAME EXCLUSIONS"
    z1 = {"reading": word, "detail": new_summary["verdict_detail"], "gate_k30": new_summary["gate_k30"],
          "reader": "scripts/novel_schema_summary.py, unchanged"}
    if comparable:
        z2 = z.agreement_reading(new_summary, banked_summary)
        z3 = z.spread_reading(new_arms, banked_arms)
    else:
        suspended = ("NOT COMPARABLE: Z4 read DIFFERENT EXCLUSIONS, so the two runs scored "
                     "different document sets; the citation rule is suspended and the discrepancy "
                     "goes to CORRECTIONS.md before any number moves")
        z2 = {"reading": "NOT COMPARABLE", "detail": suspended,
              "new_mean_delta": new_summary["gate_k30"].get("mean_delta"),
              "banked_mean_delta": banked_summary["gate_k30"].get("mean_delta")}
        z3 = {"reading": "NOT COMPARABLE", "detail": suspended}
    preregistered = {
        "Z1": {"predicted": "GO", "observed": z1["reading"], "holds": z1["reading"] == "GO"},
        "Z2": {"predicted": "AGREES", "observed": z2["reading"], "holds": z2["reading"] == "AGREES"},
        "Z3": {"predicted": f"max per-arm spread <= {z.SPREAD_PREDICTION}", "observed": z3.get("max_abs_diff"),
               "holds": bool(z3.get("prediction_holds", False))},
        "Z4": {"predicted": "SAME EXCLUSIONS", "observed": z4["reading"],
               "holds": z4["reading"] == "SAME EXCLUSIONS"},
    }
    headline = {"banked": banked_summary["gate_k30"].get("mean_delta"),
                "primary_verifiable": new_summary["gate_k30"].get("mean_delta"),
                "rule": "...",
                "quote": new_summary["gate_k30"].get("mean_delta") if (word == "GO" and comparable) else None}
    reading = {
        "protocol": z.PROTOCOL, "spec": z.SPEC, "date": z.DATE,
        "replicates": f"novel_schema_summary_{z.BANKED_DATE}.json and its six arms",
        "environment": {"torch": "not imported in this test"},
        "arms": {z.artifact_name(s, a): {k: new_arms[(z.K, s, a)].get(k) for k in
                                          ("mean_micro_f1", "scored", "no_completion", "restarts",
                                           "adapt_seconds", "compute_seconds")}
                 for s in z.SEEDS for a in z.ARMS},
        "Z1_gate": z1, "Z2_agreement": z2, "Z3_spread": z3, "Z4_attrition": z4,
        "preregistered": preregistered, "headline": headline, "verifiability": "...",
    }
    return json.loads(json.dumps(reading))  # exactly what a reader of the file gets back


CASES = {
    # name: (_case kwargs, expected outcome, expected Z1 word)
    "agrees": ({"delta": 0.47}, "AGREES", "GO"),
    "moved": ({"delta": 0.30}, "MOVED", "GO"),
    "agrees-wider-spread": ({"delta": 0.49}, "AGREES", "GO"),
    "pivot": ({"delta": 0.02, "jitter": 0.03}, "WITHDRAWN", "PIVOT"),
    "uninformative": ({"delta": 0.47, "kshot_level": 0.05}, "WITHDRAWN", "UNINFORMATIVE"),
    "different-exclusions-go": ({"delta": 0.47, "seed2_over_cap": tuple(range(39, 60))}, "NOT_COMPARABLE", "GO"),
    "different-exclusions-pivot": ({"delta": 0.02, "jitter": 0.03, "seed2_over_cap": tuple(range(39, 60))},
                                   "NOT_COMPARABLE", "PIVOT"),
    "agrees-with-empty-pools": ({"delta": 0.47, "empty": (3, 4)}, "AGREES", "GO"),
}


def _reading(name):
    new_arms, banked_arms, new_summary, banked_summary = _case(**CASES[name][0])
    return build_reading(new_arms, banked_arms, new_summary, banked_summary), new_summary, banked_summary


def _stage(tmp_path, name):
    reading, new_summary, banked_summary = _reading(name)
    (tmp_path / pub.READING_NAME).write_text(json.dumps(reading))
    (tmp_path / pub.NEW_SUMMARY_NAME).write_text(json.dumps(new_summary))
    (tmp_path / pub.BANKED_SUMMARY_NAME).write_text(json.dumps(banked_summary))
    return reading, new_summary, banked_summary


@pytest.fixture(autouse=True)
def frozen_pages(monkeypatch):
    """Every test reads the frozen excerpts, never the live pages."""
    monkeypatch.setattr(pub, "FIXTURE_MODE", True)


def _page(rel: str) -> str:
    return pub.page_path(rel).read_text(encoding="utf-8")


@pytest.fixture
def offline(monkeypatch):
    """No git, no run log, no arms: those checks have their own tests below."""
    monkeypatch.setattr(pub, "check_code_pins", lambda pins=None: [])
    monkeypatch.setattr(pub, "check_run_log", lambda directory: None)
    monkeypatch.setattr(pub, "check_primary", lambda directory, banked_dir: None)


# -- the shape is read()'s ----------------------------------------------------


def _dict_keys_assigned_in_read() -> dict:
    tree = ast.parse((REPO / "scripts" / "novel_schema_rerun.py").read_text())
    read = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "read")
    found: dict = {}
    for node in ast.walk(read):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.setdefault(target.id, []).append(node.value)
    return found


def _keys(node: ast.Dict) -> set:
    return {k.value for k in node.keys if isinstance(k, ast.Constant)}


def test_the_synthetic_reading_has_exactly_the_keys_read_writes():
    assigned = _dict_keys_assigned_in_read()
    for name in ("agrees", "different-exclusions-go"):
        reading, _, _ = _reading(name)
        report = assigned["report"][0]
        assert _keys(report) == set(reading)
        assert _keys(assigned["z1"][0]) == set(reading["Z1_gate"])
        assert _keys(assigned["headline"][0]) == set(reading["headline"])
        prereg = assigned["preregistered"][0]
        assert _keys(prereg) == set(reading["preregistered"])
        for value in prereg.values:
            assert _keys(value) == set(reading["preregistered"]["Z1"])
        arms = next(v for k, v in zip(report.keys, report.values) if k.value == "arms")
        assert isinstance(arms, ast.DictComp) and _keys(arms.value) == set(next(iter(reading["arms"].values())))
        if name == "different-exclusions-go":
            assert _keys(assigned["z2"][0]) == set(reading["Z2_agreement"])
            assert _keys(assigned["z3"][0]) == set(reading["Z3_spread"])


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_outcome_the_runner_can_emit_is_recognised(name):
    reading, new_summary, banked_summary = _reading(name)
    _, outcome, word = CASES[name]
    assert reading["Z1_gate"]["reading"] == word
    assert pub.check_consistency(reading, new_summary, banked_summary) == outcome


def test_headline_format_is_the_citation_tests():
    tree = ast.parse(CITATION_TEST.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "headline_string")
    scope: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(CITATION_TEST), "exec"), scope)  # noqa: S102
    for x in (0.4648, 0.4712, -0.0049, 0.4, 0.0, 0.30551):
        assert pub.pts(x) == scope["headline_string"](x)
    assert pub.pts(pub.BANKED_MEAN) == pub.BANKED_PTS


def test_quoted_protocol_sentences_and_b93_clauses_are_verbatim():
    protocol = _norm(pub.PROTOCOL_PATH.read_text(encoding="utf-8"))
    for quote in pub.PROTOCOL_QUOTES:
        assert _norm(quote).lower() in protocol.lower(), quote
    source = inspect.getsource(z.spread_reading)
    for clause in pub.B93_CLAUSES:
        assert clause in source


# -- refusals -----------------------------------------------------------------


def test_refuses_and_prints_nothing_without_the_reading(tmp_path, capsys):
    assert pub.main(["--dir", str(tmp_path), "--banked-dir", str(tmp_path)]) == 2
    out, err = capsys.readouterr()
    assert out == "" and "REFUSED" in err and pub.READING_NAME in err


def test_refuses_without_the_summaries(tmp_path, capsys, offline):
    _stage(tmp_path, "agrees")
    (tmp_path / pub.NEW_SUMMARY_NAME).unlink()
    assert pub.main(["--dir", str(tmp_path), "--banked-dir", str(tmp_path)]) == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("key,value", [("protocol", "docs/research/ADDENDUM_Z_PROTOCOL.md"),
                                       ("spec", "something else"), ("date", "2026-08-12")])
def test_refuses_a_reading_without_the_runners_identity(tmp_path, capsys, offline, key, value):
    reading, _, _ = _stage(tmp_path, "agrees")
    reading[key] = value
    (tmp_path / pub.READING_NAME).write_text(json.dumps(reading))
    assert pub.main(["--dir", str(tmp_path), "--banked-dir", str(tmp_path)]) == 2
    out, err = capsys.readouterr()
    assert out == "" and key in err


def test_protocol_anchor_holds_today_and_refuses_an_edited_protocol(tmp_path, monkeypatch):
    assert pub.check_protocol_anchor() == hashlib.sha256(pub.PROTOCOL_PATH.read_bytes()).hexdigest()
    copy = tmp_path / pub.PROTOCOL_PATH.name
    shutil.copy(pub.PROTOCOL_PATH, copy)
    proof = pub.PROTOCOL_PATH.with_name(f"{pub.PROTOCOL_PATH.name}.{pub.PROTOCOL_STAMP}.ots")
    shutil.copy(proof, tmp_path / proof.name)
    monkeypatch.setattr(pub, "PROTOCOL_PATH", copy)
    assert pub.check_protocol_anchor()
    copy.write_bytes(copy.read_bytes().replace(b"Z2 = AGREES", b"Z2 = GO BUT MOVED"))
    with pytest.raises(pub.Refused, match="no longer hashes"):
        pub.check_protocol_anchor()


def _pins(**changed):
    out = {}
    for rel in pub.runlog.PINNED:
        then, now = changed.get(rel, ("a" * 64, "a" * 64))
        out[rel] = {"at_freeze_commit": then, "now": now, "unchanged_since_freeze": then == now}
    return out


def test_code_pins_refuse_any_change_but_the_named_launcher_fix():
    launcher = "scripts/run_addendum_z.sh"
    assert pub.check_code_pins(_pins()) == []
    assert pub.check_code_pins(_pins(**{launcher: pub.KNOWN_CHANGES[launcher]})) == [launcher]
    with pytest.raises(pub.Refused, match="run_addendum_z.sh"):
        pub.check_code_pins(_pins(**{launcher: (pub.KNOWN_CHANGES[launcher][0], "b" * 64)}))
    with pytest.raises(pub.Refused, match="novel_schema_rerun.py"):
        pub.check_code_pins(_pins(**{"scripts/novel_schema_rerun.py": ("a" * 64, "c" * 64)}))
    with pytest.raises(pub.Refused, match="PINNED"):
        pub.check_code_pins({"scripts/novel_schema_rerun.py": _pins()["scripts/novel_schema_rerun.py"]})


def _has_freeze_commit() -> bool:
    try:
        return subprocess.run(["git", "cat-file", "-e", f"{pub.runlog.FREEZE_COMMIT}^{{commit}}"],
                              cwd=REPO, capture_output=True, check=False).returncode == 0
    except OSError:
        return False


@pytest.mark.skipif(not _has_freeze_commit(), reason="this checkout lacks the freeze commit's history")
def test_this_checkout_passes_the_code_pins():
    assert set(pub.check_code_pins()) <= set(pub.KNOWN_CHANGES)


def test_refuses_when_the_run_log_shows_a_redraw(tmp_path, monkeypatch):
    log = tmp_path / "addendum_z_run_log.json"
    monkeypatch.setattr(pub.runlog, "OUT", log)
    with pytest.raises(pub.Refused, match="redraw would be undetectable"):
        pub.check_run_log(tmp_path)
    log.write_text("{}")
    monkeypatch.setattr(pub.runlog, "verify", lambda d: ["2026-09-23T03:00:00Z arm: adapter redrawn"])
    with pytest.raises(pub.Refused, match="adapter redrawn"):
        pub.check_run_log(tmp_path)


def test_refuses_when_the_strangers_verifier_does_not_pass(tmp_path, monkeypatch):
    import types

    fake = types.ModuleType("verify_z_from_primary")
    fake.main = lambda argv: (print("seed 1 adapted: 3 stored scores do not re-derive"), 1)[1]
    monkeypatch.setitem(sys.modules, "verify_z_from_primary", fake)
    with pytest.raises(pub.Refused, match="exited 1: seed 1 adapted"):
        pub.check_primary(tmp_path, tmp_path)
    fake.main = lambda argv: 0
    pub.check_primary(tmp_path, tmp_path)


@pytest.mark.parametrize("tamper", ["quote", "z2", "z3word", "z4word", "summary"])
def test_refuses_a_reading_that_contradicts_itself_or_its_summaries(tamper):
    reading, new_summary, banked_summary = _reading("agrees")
    if tamper == "quote":
        reading["headline"]["quote"] = None
    elif tamper == "z2":
        reading["Z2_agreement"]["reading"] = "GO BUT MOVED"
    elif tamper == "z3word":
        reading["Z3_spread"]["reading"] = "SPREAD LARGER THAN PREDICTED"
    elif tamper == "z4word":
        reading["Z4_attrition"]["reading"] = "DIFFERENT EXCLUSIONS"
    else:
        new_summary["VERDICT"] = "PIVOT"
    with pytest.raises(pub.Refused):
        pub.check_consistency(reading, new_summary, banked_summary)


def test_refuses_an_unfilled_or_unlicensed_placeholder():
    reading, new_summary, _ = _reading("pivot")
    vals = pub.values(reading, new_summary)
    assert vals["NEW_PTS"] is None and vals["MOVE_PHRASE"] is None  # a withdrawn headline has no new number
    with pytest.raises(pub.Refused, match="no value"):
        pub.fill("GO at {NEW_PTS}", vals, "test")
    with pytest.raises(pub.Refused, match="unknown placeholder"):
        pub.fill("{NOT_A_FIELD}", vals, "test")
    assert pub.fill("seeds {1,2,3}: {Z1_WORD}", vals, "test") == "seeds {1,2,3}: PIVOT"


# -- the non-flattering rule ----------------------------------------------------


def test_every_staged_template_obeys_the_frozen_citation_rule():
    assert pub.all_template_problems() == []


@pytest.mark.parametrize("outcome,text,needle", [
    ("WITHDRAWN", "The gate decided GO at +46.5.", "without 'withdrawn'"),
    ("WITHDRAWN", "Withdrawn. Now {NEW_PTS}.", "cannot quote"),
    ("AGREES", "Adaptation lifted extraction by +46.5.", "not marked banked"),
    ("AGREES", "The re-run confirmed {NEW_PTS}, banked +46.5.", "flattering"),
    ("MOVED", "Now {NEW_PTS}; banked +46.5 is history.", "movement"),
    ("NOT_COMPARABLE", "GO at +46.5.", "NOT COMPARABLE"),
    ("NOT_COMPARABLE", "The re-run reads {NEW_PTS}, NOT COMPARABLE.", "nothing moves"),
    ("AGREES", "Re-run pending: banked +46.5.", "stale"),
])
def test_the_rule_checker_catches_what_it_is_for(outcome, text, needle):
    assert any(needle in p for p in pub.template_problems("probe", outcome, text))


def test_every_anchor_is_found_exactly_once_in_the_frozen_excerpts():
    assert pub.check_anchors(scan=[]) == []
    for site in pub.ALL_SITES:
        assert pub.page_path(site["path"]).exists(), site["path"]


def _copy_fixtures(tmp_path, monkeypatch):
    public, internal = tmp_path / "public", tmp_path / "internal"
    shutil.copytree(pub.FIXTURES, public)
    if pub.INTERNAL_FIXTURES.exists():
        shutil.copytree(pub.INTERNAL_FIXTURES, internal)
    monkeypatch.setattr(pub, "FIXTURES", public)
    monkeypatch.setattr(pub, "INTERNAL_FIXTURES", internal)
    return public


def test_the_anchor_check_reports_a_missing_anchor_an_uncovered_quote_and_a_dated_rewrite(
        tmp_path, monkeypatch):
    public = _copy_fixtures(tmp_path, monkeypatch)
    roadmap = public / "ROADMAP.md.fixture"
    roadmap.write_text(roadmap.read_text().replace("- **Addendum Z (preregistered", "- **Addendum Z (renamed"))
    evidence = public / "EVIDENCE.md.fixture"
    evidence.write_text(evidence.read_text() + "\nA new sentence that quotes +46.5 as a result.\n")
    challenges = public / "CHALLENGES.md.fixture"
    challenges.write_text("## Update - 2026-10-01: a dated record\n\n" + challenges.read_text())
    problems = pub.check_anchors(scan=["EVIDENCE.md"])
    assert any("[roadmap-z]" in p and "not found" in p for p in problems), problems
    assert any(p.startswith("EVIDENCE.md:") and "no site rewrites" in p for p in problems), problems
    assert any("[challenges-format-table]" in p and "dated section" in p for p in problems), problems


def test_anchors_tolerate_rewrapping_but_not_rewording():
    page = "a protocol anchored before\n  the first arm, and then"
    assert pub.locate(page, "replace", "protocol anchored before the first arm,", "protocol anchored before "
                      "the first arm,") is not None
    assert pub.locate(page, "replace", "protocol anchored after the first arm,", "x") is None
    assert pub.locate("x\nx\n", "replace", "x", "x") is None  # ambiguous


def _sentences_with_banked_number(text: str) -> list[str]:
    return [u for u in pub._units(text) if pub.BANKED_PTS in u]


@pytest.mark.parametrize("name", sorted(CASES))
def test_printed_text_applied_to_the_pages_obeys_the_frozen_citation_rule(name, tmp_path, capsys, offline):
    reading, new_summary, _ = _stage(tmp_path, name)
    _, outcome, word = CASES[name]
    before = {s["path"]: pub.page_path(s["path"]).read_bytes() for s in pub.ALL_SITES}
    assert pub.main(["--dir", str(tmp_path), "--banked-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert f"outcome: {outcome}" in out and not pub.PLACEHOLDER.search(out)
    assert "ANCHOR NOT FOUND" not in out
    assert {p: pub.page_path(p).read_bytes() for p in before} == before  # printed, never written

    vals = pub.values(reading, new_summary)
    pages = {}
    for entry in sorted(pub.render(outcome, vals), key=lambda e: -(e["span"] or (0, 0))[0]):
        if entry["new"] is None or entry["span"] is None:
            continue
        page = pages.get(entry["path"], _page(entry["path"]))
        pages[entry["path"]] = pub.apply(page, entry)

    # the landing half of tests/test_addendum_z_citation.py, on the applied pages
    comparable = outcome != "NOT_COMPARABLE"
    for name_ in HEADLINE_PAGES:
        text = pages[name_]
        assert "Addendum Z" in text and "re-run pending" not in text.lower(), name_
        if not comparable:
            assert "NOT COMPARABLE" in text or "DIFFERENT EXCLUSIONS" in text, name_
        elif word != "GO":
            assert reading["headline"]["quote"] is None and "withdrawn" in text.lower(), name_
        else:
            assert pub.pts(reading["headline"]["quote"]) in text, name_
            assert "+46.5" not in text or "banked" in text.lower(), name_
    corrections = pages["CORRECTIONS.md"]
    if comparable and word == "GO":
        assert pub.pts(reading["headline"]["quote"]) in corrections
    if comparable and word != "GO":
        assert "withdrawn" in corrections.lower() and "+46.5" in corrections

    # stricter than the landing test, on every live page this script rewrites:
    # the ledger and the frozen spec keep their history, banners mark records
    banners = {s["path"] for s in pub.ALL_SITES if s["kind"] in ("prepend", "insert_before")}
    live = {s["path"] for s in pub.ALL_SITES} - banners - {
        "CORRECTIONS.md", "docs/research/ENTERPRISE_EVAL_SPEC.md", "tests/test_verification_coverage.py"}
    for path in sorted(live):
        text = pages.get(path) or _page(path)
        assert "re-run pending" not in text.lower() and "**Pending; no number exists yet.**" not in text, path
        for unit in _sentences_with_banked_number(text):
            low = unit.lower()
            if outcome == "WITHDRAWN":
                assert "withdraw" in low, (path, unit[:160])
            elif outcome in ("AGREES", "MOVED"):
                assert "banked" in low or "history" in low, (path, unit[:160])
        if outcome == "MOVED" and vals["NEW_PTS"] in text:
            assert "GO BUT MOVED" in text or vals["MOVE_PHRASE"] in text, path
    for site in pub.ALL_SITES:
        if site["kind"] in ("prepend", "insert_before"):
            banner = pages[site["path"]][:4000] if site["kind"] == "prepend" else pages[site["path"]]
            assert "Addendum Z landed" in banner, site["path"]
            if outcome == "WITHDRAWN":
                assert "withdrawn" in banner, site["path"]


def test_a_withdrawn_reading_quotes_the_protocols_own_words_and_no_new_headline(tmp_path, capsys, offline):
    reading, new_summary, _ = _stage(tmp_path, "pivot")
    assert pub.main(["--dir", str(tmp_path), "--banked-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert pub.P_DISAGREE in out and pub.P_FIRST in out
    assert "Z1 predicted GO, observed PIVOT (did not hold)" in out
    vals = pub.values(reading, new_summary)
    assert vals["NEW_PTS"] is None
    # the re-run's own mean is reported once, in the Z row, as the reader's PIVOT
    assert out.count(vals["RERUN_MEAN_SENTENCE"]) == 1


def test_a_drifted_anchor_is_printed_by_hand_and_exits_3(capsys):
    reading, new_summary, _ = _reading("agrees")
    vals = pub.values(reading, new_summary)
    entries = pub.render("AGREES", vals, only="roadmap-z")
    assert entries[0]["span"] is not None
    entries[0]["span"] = None
    assert pub.print_report("AGREES", entries, []) == 3
    assert "ANCHOR NOT FOUND" in capsys.readouterr().out


def test_templates_mode_prints_no_number_and_needs_no_reading(capsys):
    assert pub.main(["--templates", "WITHDRAWN"]) == 0
    out = capsys.readouterr().out
    assert "{Z1_WORD}" in out and "{NEW_PTS}" not in out


def test_help_names_the_anchor_check_and_the_launcher_digests(capsys):
    with pytest.raises(SystemExit):
        pub.main(["--help"])
    text = capsys.readouterr().out
    assert "--check-anchors" in text and "not a CI test" in text
    before, after = pub.KNOWN_CHANGES["scripts/run_addendum_z.sh"]
    assert before in text and after in text and "only at its exact digests" in text


def test_internal_sites_load_from_their_data_file_when_present():
    internal = [s for s in pub.ALL_SITES if s.get("internal")]
    if not pub.INTERNAL_SITES.exists():
        assert internal == []  # the public cut: no internal page is named anywhere
        return
    assert internal and all(s["path"] in pub.INTERNAL_PATHS for s in internal)
    assert {s["path"] for s in pub.ALL_SITES if not s.get("internal")}.isdisjoint(pub.INTERNAL_PATHS)


def test_nothing_this_tool_exports_trips_the_public_leak_gate():
    gate = REPO / "scripts" / "leak_gate.sh"
    if not gate.exists():
        pytest.skip("the leak gate is not in this cut")
    pattern = re.search(r"GATE_RE='([^']+)'", gate.read_text()).group(1)
    exported = [REPO / "scripts" / "publish_z_reading.py", pathlib.Path(__file__)]
    exported += sorted(pub.FIXTURES.rglob("*"))
    for path in exported:
        if path.is_file():
            hit = re.search(pattern, path.read_text(encoding="utf-8"), re.IGNORECASE)
            assert hit is None, (path.name, hit.group(0) if hit else None)


def test_imports_without_torch():
    code = ("import sys; sys.modules['torch'] = None; sys.modules['transformers'] = None; "
            f"sys.path.insert(0, {str(REPO / 'scripts')!r}); import publish_z_reading as p; "
            "assert p.all_template_problems() == []")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, check=False)
    assert result.returncode == 0, result.stderr
