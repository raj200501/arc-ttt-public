"""Addendum Z: the readings at their frozen boundaries, the traced decode
against the shipped one, and the arm runner end to end on a fake model —
resume included — so the artifact the unchanged reader consumes and the
coverage classifier grades PRIMARY is pinned before a single real arm
runs."""
import importlib.util
import json
import pathlib
import shutil
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rerun = _load("novel_schema_rerun")
summary = _load("novel_schema_summary")
coverage = _load("verification_coverage")
BANKED = REPO / "experiments"


# -- the traced decode is the shipped decode ----------------------------------


class _PoolPredictor:
    def __init__(self, texts, lps):
        self.texts, self.lps = texts, lps
        self.lp_calls = 0

    def predict_text(self, task, index, samples, include_demos=True):
        return list(self.texts)

    def log_probabilities_text(self, task, index, outputs, include_demos=True):
        self.lp_calls += 1
        return [self.lps[o] for o in outputs]


def test_traced_decode_selects_what_predict_text_voted_selects():
    from arcttt.text_ttt import predict_text_voted

    texts = ['{"a": 1}', '{"a":1}', '{"a": 2}', 'not json', '{"a": 2}']
    lps = {'{"a": 1}': -0.5, '{"a":1}': -0.4, '{"a": 2}': -0.1, 'not json': -3.0}
    shipped = predict_text_voted(_PoolPredictor(texts, lps), None, 0, 5)
    traced_predictor = _PoolPredictor(texts, lps)
    selected, trace = rerun.voted_with_trace(traced_predictor, None, 0, 5)
    assert selected == shipped
    assert trace["completions"] == texts
    assert traced_predictor.lp_calls == 4  # once per DISTINCT completion
    assert dict(map(tuple, trace["distinct_log_probabilities"])) == lps
    assert {c["key"] for c in trace["candidates"]} == {'{"a":1}', '{"a":2}', "not json"}
    assert sum(c["found_count"] for c in trace["candidates"]) == 5


def test_traced_decode_records_the_lp_that_decides_a_count_tie():
    from arcttt.text_ttt import predict_text_voted

    # two pools of two; exp(mean lp) breaks the tie toward {"b":1}
    texts = ['{"a": 1}', '{"b": 1}', '{"a":1}', '{"b":1}']
    lps = {'{"a": 1}': -2.0, '{"b": 1}': -0.2, '{"a":1}': -2.0, '{"b":1}': -0.3}
    selected, trace = rerun.voted_with_trace(_PoolPredictor(texts, lps), None, 0, 4)
    assert selected == predict_text_voted(_PoolPredictor(texts, lps), None, 0, 4) == '{"b": 1}'
    by_key = {c["key"]: c for c in trace["candidates"]}
    assert by_key['{"b":1}']["found_count"] == 2 and by_key['{"b":1}']["mean_log_probability"] == -0.25
    assert by_key['{"a":1}']["mean_log_probability"] == -2.0


def test_traced_decode_with_empty_pool_returns_none():
    selected, trace = rerun.voted_with_trace(_PoolPredictor([], {}), None, 0, 5)
    assert selected is None and trace["completions"] == []


# -- readings at the boundaries ------------------------------------------------


def _summary(word, mean):
    return {"VERDICT": word, "verdict_detail": "", "gate_k30": {"mean_delta": mean}}


def test_z2_agrees_at_exactly_the_tolerance_and_moves_just_past_it():
    banked = _summary("GO", 0.4648)
    assert rerun.agreement_reading(_summary("GO", 0.4648 + 0.05), banked)["reading"] == "AGREES"
    assert rerun.agreement_reading(_summary("GO", 0.4648 - 0.05), banked)["reading"] == "AGREES"
    assert rerun.agreement_reading(_summary("GO", 0.4648 + 0.0501), banked)["reading"] == "GO BUT MOVED"
    assert rerun.agreement_reading(_summary("GO", 0.4648 - 0.0501), banked)["reading"] == "GO BUT MOVED"


def test_z2_a_different_verdict_disagrees_and_undecidable_withholds():
    banked = _summary("GO", 0.4648)
    assert rerun.agreement_reading(_summary("PIVOT", 0.04), banked)["reading"] == "DISAGREES"
    assert rerun.agreement_reading(_summary("UNINFORMATIVE", 0.9), banked)["reading"] == "DISAGREES"
    assert rerun.agreement_reading(_summary("UNDECIDABLE", None), banked)["reading"] == "WITHHELD"
    with pytest.raises(ValueError):  # the banked verdict is GO by construction
        rerun.agreement_reading(_summary("PIVOT", 0.0), _summary("PIVOT", 0.0))


def _arms(means, excluded=None, reason=rerun.OVER_CAP):
    excluded = excluded or {}
    arms = {}
    for seed in rerun.SEEDS:
        for arm in rerun.ARMS:
            gone = set(excluded.get((seed, arm), ()))
            results = [
                {"index": i, "error": "no completion"} if i in gone
                else {"index": i, "micro_f1": 0.5}
                for i in range(6)
            ]
            predictions = [
                {"index": i, "error": "no completion", "reason": reason} if i in gone
                else {"index": i, "prediction": "{}", "micro_f1_raw": 0.5}
                for i in range(6)
            ]
            arms[(30, seed, arm)] = {"mean_micro_f1": means[(seed, arm)], "results": results,
                                     "predictions": predictions}
    return arms


def _banked_arms(means, excluded=None):
    arms = _arms(means, excluded)
    for record in arms.values():
        del record["predictions"]  # the banked arms carry no reasons
    return arms


def test_z3_spread_reports_the_largest_arm_the_prediction_and_the_b93_replacement():
    old = {(s, a): 0.5 for s in rerun.SEEDS for a in rerun.ARMS}
    new = dict(old)
    new[(2, "kshot")] = 0.53  # exactly the prediction
    reading = rerun.spread_reading(_arms(new), _banked_arms(old))
    assert reading["max_abs_diff"] == 0.03 and reading["max_at"] == {"seed": 2, "arm": "kshot"}
    assert reading["prediction_holds"] and reading["reading"] == "WITHIN PREDICTED SPREAD"
    assert reading["replaces_b93"] is True
    new[(3, "adapted")] = 0.5301
    reading = rerun.spread_reading(_arms(new), _banked_arms(old))
    assert not reading["prediction_holds"] and reading["reading"] == "SPREAD LARGER THAN PREDICTED"
    # the B.9.3 figure is replaced only by something larger
    new = dict(old)
    new[(1, "adapted")] = 0.5097
    assert rerun.spread_reading(_arms(new), _banked_arms(old))["replaces_b93"] is False
    new[(1, "adapted")] = 0.5099
    assert rerun.spread_reading(_arms(new), _banked_arms(old))["replaces_b93"] is True


def test_z4_attrition_compares_the_over_cap_documents_per_arm_and_ignores_empty_pools():
    means = {(s, a): 0.5 for s in rerun.SEEDS for a in rerun.ARMS}
    banked = _banked_arms(means, {(2, "adapted"): (0, 3), (2, "kshot"): (0, 3)})
    same = rerun.attrition_reading(_arms(means, {(2, "adapted"): (0, 3), (2, "kshot"): (0, 3)}), banked)
    assert same["reading"] == "SAME EXCLUSIONS"
    other = rerun.attrition_reading(_arms(means, {(2, "adapted"): (0, 3), (2, "kshot"): (0, 4)}), banked)
    assert other["reading"] == "DIFFERENT EXCLUSIONS"
    bad = [r for r in other["per_arm"] if not r["same"]]
    assert bad == [{"seed": 2, "arm": "kshot", "over_cap_new": [0, 4], "over_cap_banked": [0, 3],
                    "empty_pool_new": [], "same": False}]
    # an empty pool is sampling attrition, not an over-cap exclusion
    new = _arms(means, {(2, "adapted"): (0, 3), (2, "kshot"): (0, 3)})
    new[(30, 1, "kshot")]["predictions"][5] = {"index": 5, "error": "no completion", "reason": rerun.EMPTY_POOL}
    new[(30, 1, "kshot")]["results"][5] = {"index": 5, "error": "no completion"}
    reading = rerun.attrition_reading(new, banked)
    assert reading["reading"] == "SAME EXCLUSIONS"
    assert [r for r in reading["per_arm"] if r["seed"] == 1 and r["arm"] == "kshot"][0]["empty_pool_new"] == [5]


# -- read(): staged end to end -------------------------------------------------


def _stage(tmp_path, means=None, excluded=None, drop=None, protocol=rerun.PROTOCOL):
    """Six fake 2026-09-23 arms beside real copies of the banked run."""

    out, banked = tmp_path / "out", tmp_path / "banked"
    out.mkdir()
    banked.mkdir()
    for path in BANKED.glob("novel_schema_0.5b_k30_seed*_2026-08-12.json"):
        shutil.copy(path, banked / path.name)
    shutil.copy(BANKED / "novel_schema_summary_2026-08-12.json", banked / "novel_schema_summary_2026-08-12.json")
    banked_arms = summary.load(banked, "2026-08-12")
    for seed in rerun.SEEDS:
        for arm in rerun.ARMS:
            if drop == (seed, arm):
                continue
            old = banked_arms[(30, seed, arm)]
            gone = set(excluded.get((seed, arm), rerun.excluded_indices(old))) if excluded is not None \
                else set(rerun.excluded_indices(old))
            rows = old["results"]
            results, predictions = [], []
            for row in rows:
                i = row["index"]
                if i in gone:
                    results.append({"index": i, "error": "no completion"})
                    predictions.append({"index": i, "error": "no completion", "reason": rerun.OVER_CAP})
                    continue
                f1 = (means or {}).get((seed, arm), row.get("micro_f1", 0.0))
                results.append({"index": i, "valid_json": True, "exact_match": False, "micro_f1": f1})
                predictions.append({"index": i, "prediction": "{}", "micro_f1_raw": f1})
            scored = [r["micro_f1"] for r in results if "micro_f1" in r]
            record = {
                "protocol": protocol, "rung": "0.5b", "k": 30, "seed": seed, "arm": arm,
                "eval_n": 60, "device": "cpu", "dtype": "torch.float32",
                "mean_micro_f1": round(sum(scored) / len(scored), 4), "scored": len(scored),
                "no_completion": len(rows) - len(scored), "restarts": 0, "adapt_seconds": 1.0,
                "compute_seconds": 2.0, "results": results, "predictions": predictions,
            }
            (out / rerun.artifact_name(seed, arm)).write_text(json.dumps(record))
    return out, banked


def test_read_withholds_and_writes_nothing_with_five_of_six_arms(tmp_path, capsys):
    out, banked = _stage(tmp_path, drop=(3, "kshot"))
    assert rerun.read(out, banked) == 2
    assert "WITHHELD" in capsys.readouterr().out
    assert len(list(out.glob("*.json"))) == 5  # the five arms and nothing else


def test_read_withholds_when_an_arm_is_not_an_addendum_z_arm(tmp_path):
    out, banked = _stage(tmp_path, protocol="something else")
    assert rerun.read(out, banked) == 2
    assert not (out / "novel_schema_summary_2026-09-23.json").exists()


def test_read_withholds_when_the_banked_side_is_missing(tmp_path):
    out, banked = _stage(tmp_path)
    (banked / "novel_schema_summary_2026-08-12.json").unlink()
    assert rerun.read(out, banked) == 2
    assert not (out / "novel_schema_summary_2026-09-23.json").exists()


def test_read_happy_path_writes_the_summary_and_the_reading_together(tmp_path):
    out, banked = _stage(tmp_path)  # the banked scores replayed exactly
    assert rerun.read(out, banked) == 0
    new_summary = json.loads((out / "novel_schema_summary_2026-09-23.json").read_text())
    reading = json.loads((out / "novel_schema_rerun_2026-09-23.json").read_text())
    assert new_summary["VERDICT"] == "GO" and new_summary["gate_k30"]["mean_delta"] == 0.4648
    assert reading["Z1_gate"]["reading"] == "GO"
    assert reading["Z2_agreement"]["reading"] == "AGREES" and reading["Z2_agreement"]["moved_by"] == 0.0
    assert reading["Z3_spread"]["max_abs_diff"] == 0.0 and reading["Z3_spread"]["replaces_b93"] is False
    assert reading["Z4_attrition"]["reading"] == "SAME EXCLUSIONS"
    assert all(v["holds"] for v in reading["preregistered"].values())
    assert reading["headline"] == {
        "banked": 0.4648, "primary_verifiable": 0.4648, "quote": 0.4648, "rule": reading["headline"]["rule"]}
    # the reading file is AGGREGATE by the coverage rule; the arms are PRIMARY
    assert coverage._classify(reading)[0] != "PRIMARY"
    assert coverage._classify(new_summary)[0] != "PRIMARY"
    assert coverage._classify(json.loads((out / rerun.artifact_name(1, "adapted")).read_text()))[0] == "PRIMARY"
    # the unchanged reader's glob sees exactly the six arms, not the two outputs
    assert len(summary.load(out, rerun.DATE)) == 6


def test_read_different_exclusions_suspends_z2_z3_and_the_quote(tmp_path):
    excluded = {(2, "adapted"): set(range(21)), (2, "kshot"): set(range(21))}  # 21 not 22
    out, banked = _stage(tmp_path, excluded=excluded)
    assert rerun.read(out, banked) == 0
    reading = json.loads((out / "novel_schema_rerun_2026-09-23.json").read_text())
    assert reading["Z4_attrition"]["reading"] == "DIFFERENT EXCLUSIONS"
    assert reading["Z2_agreement"]["reading"] == "NOT COMPARABLE"
    assert reading["Z3_spread"]["reading"] == "NOT COMPARABLE"
    assert reading["headline"]["quote"] is None
    assert reading["preregistered"]["Z4"]["holds"] is False and reading["preregistered"]["Z2"]["holds"] is False


def test_read_pivot_disagrees_and_withdraws_the_quote(tmp_path):
    means = {(s, "adapted"): 0.5 for s in rerun.SEEDS}
    means.update({(s, "kshot"): 0.5 for s in rerun.SEEDS})
    out, banked = _stage(tmp_path, means=means)
    assert rerun.read(out, banked) == 0
    reading = json.loads((out / "novel_schema_rerun_2026-09-23.json").read_text())
    assert reading["Z1_gate"]["reading"] == "PIVOT"
    assert reading["Z2_agreement"]["reading"] == "DISAGREES"
    assert reading["headline"]["quote"] is None


# -- the runner, end to end on a fake model, resume included ------------------


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_lora_a = torch.nn.Parameter(torch.ones(2, 2))

    def named_parameters(self, *a, **k):  # the runner keys on "lora_" in the name
        return [("layer.lora_a", self.layer_lora_a)]


class _FakeTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, chat, add_generation_prompt=True, return_tensors="pt"):
        return torch.zeros(1, 9000, dtype=torch.long)


class _FakePredictor:
    calls = {"predict": 0, "adapt": 0}
    over_cap = {1}
    empty_pool = set()

    def __init__(self, model, tokenizer, config, device):
        self.model, self.tokenizer, self.config = model, tokenizer, config

    def adapt_text(self, task, shuffle_seeds=(None,)):
        type(self).calls["adapt"] += 1

    def _encode(self, turns, supervise_final=True):
        return (torch.zeros(1, 3), torch.zeros(1, 3))

    def _prompt_ids(self, turns):
        # the test index is the last user turn's ordinal in the fake task
        index = _FakePredictor.index_of[turns[-1].content]
        return None if index in self.over_cap else torch.tensor([[1, 2, index]])

    def predict_text(self, task, index, samples, include_demos=True):
        type(self).calls["predict"] += 1
        assert samples == rerun.POOL_SAMPLES
        if index in self.empty_pool:
            return []
        gold = task.test[index].output_text
        return [gold, gold, "{}", gold, gold]

    def log_probabilities_text(self, task, index, outputs, include_demos=True):
        return [-0.1 if o != "{}" else -2.0 for o in outputs]


@pytest.fixture
def fake_arm(monkeypatch, tmp_path):
    import arcttt.lora as lora
    import arcttt.novel_schema as novel_schema
    import arcttt.text_ttt as text_ttt

    task, schema = novel_schema.make_task(seed=1, n_train=2, n_test=4, task_id="t")
    _FakePredictor.index_of = {pair.input_text: i for i, pair in enumerate(task.test)}
    _FakePredictor.calls = {"predict": 0, "adapt": 0}
    _FakePredictor.empty_pool = set()
    monkeypatch.setattr(novel_schema, "make_task", lambda **kw: (task, schema))
    monkeypatch.setattr(text_ttt, "TextPredictor", _FakePredictor)
    monkeypatch.setattr(lora, "inject_lora", lambda *a, **k: ["layer"])
    monkeypatch.setattr(lora, "remove_lora", lambda *a, **k: None)
    work, out = tmp_path / "work", tmp_path / "out"
    out.mkdir()
    return task, work, out


def _arm(seed, arm, work, out):
    return rerun.run_arm(seed, arm, _FakeModel(), _FakeTokenizer(), work, out)


def test_run_arm_banks_a_primary_artifact_the_unchanged_reader_loads(fake_arm):
    task, work, out = fake_arm
    path = _arm(1, "adapted", work, out)
    record = json.loads(path.read_text())
    assert path.name == "novel_schema_0.5b_k30_seed1_adapted_2026-09-23.json"
    assert summary.IDENTITY.issubset(record) and record["device"] == "cpu"
    assert (30, 1, "adapted") in summary.load(out, rerun.DATE)
    assert coverage._classify(record)[0] == "PRIMARY"
    assert record["protocol"] == rerun.PROTOCOL and record["model_revision"] == rerun.MODEL_REVISION
    assert record["scored"] == 3 and record["no_completion"] == 1
    assert record["no_completion_reasons"] == {rerun.OVER_CAP: 1, rerun.EMPTY_POOL: 0}
    assert record["results"][1] == {"index": 1, "error": "no completion"}
    over = record["predictions"][1]
    assert over["reason"] == rerun.OVER_CAP and over["prompt_tokens"] == 9000 and over["cap"] == 8192
    assert len(over["prompt_sha256"]) == 64
    assert record["mean_micro_f1"] == 1.0 and record["exact_match"] == 3
    done = [p for p in record["predictions"] if "prediction" in p]
    assert len(done) == 3 and all(len(p["completions"]) == 5 for p in done)
    assert all(p["prediction"] == task.test[p["index"]].output_text for p in done)
    assert record["resumed"] is False and record["restarts"] == 0
    assert record["adapter"]["sha256"] and (work / record["adapter"]["file"]).exists()
    assert record["config"]["epochs"] == 1 and record["config"]["max_seq"] == 8192
    assert record["environment"]["torch_threads"] == torch.get_num_threads()
    assert record["environment"]["chat_template_sha256"] is not None
    assert record["compute_seconds"] >= record["decode_seconds"]
    assert _FakePredictor.calls == {"predict": 3, "adapt": 1}


def test_run_arm_resumes_from_a_partial_journal_and_the_saved_adapter(fake_arm):
    task, work, out = fake_arm
    first = _arm(1, "adapted", work, out)
    first_record = json.loads(first.read_text())
    first.unlink()
    journal = work / (rerun.ckpt_stem(1, "adapted") + "_docs.jsonl")
    lines = journal.read_text().splitlines()
    journal.write_text("\n".join(lines[:2]) + "\n")  # killed after two documents
    calls_before = dict(_FakePredictor.calls)
    second = _arm(1, "adapted", work, out)
    record = json.loads(second.read_text())
    assert _FakePredictor.calls["adapt"] == calls_before["adapt"]  # restored, not retrained
    assert _FakePredictor.calls["predict"] == calls_before["predict"] + 2  # only the missing documents
    assert record["resumed"] is True and record["restarts"] == 1
    assert record["results"] == first_record["results"]
    assert [p["index"] for p in record["predictions"]] == [0, 1, 2, 3]
    assert record["adapter"]["sha256"] == first_record["adapter"]["sha256"]
    # a third start is skip-if-exists
    assert _arm(1, "adapted", work, out) == second


def test_run_arm_survives_a_torn_journal_line_and_a_missing_meta(fake_arm):
    task, work, out = fake_arm
    first = _arm(1, "adapted", work, out)
    first.unlink()
    stem = rerun.ckpt_stem(1, "adapted")
    journal = work / f"{stem}_docs.jsonl"
    lines = journal.read_text().splitlines()
    journal.write_text("\n".join(lines[:2]) + "\n" + lines[2][:40])  # torn, no newline
    (work / f"{stem}_meta.json").unlink()  # killed between adapter and meta, as the audit feared
    record = json.loads(_arm(1, "adapted", work, out).read_text())
    assert record["scored"] == 3 and [p["index"] for p in record["predictions"]] == [0, 1, 2, 3]
    assert record["adapt_seconds"] is None  # lost with the meta, said so rather than invented
    assert record["adapter"]["sha256"]
    rows = [json.loads(l) for l in journal.read_text().splitlines()]
    assert [r["index"] for r in rows] == [0, 1, 2, 3]  # the fragment did not swallow the re-decode


def test_run_arm_retrains_when_the_saved_adapter_is_unusable(fake_arm):
    task, work, out = fake_arm
    first = _arm(1, "adapted", work, out)
    first.unlink()
    stem = rerun.ckpt_stem(1, "adapted")
    (work / f"{stem}_adapter.pt").write_bytes(b"torn")
    adapts = _FakePredictor.calls["adapt"]
    record = json.loads(_arm(1, "adapted", work, out).read_text())
    assert _FakePredictor.calls["adapt"] == adapts + 1
    assert record["restarts"] == 1 and record["adapter"]["sha256"]


def test_kshot_arm_trains_nothing_saves_no_adapter_and_counts_a_restart(fake_arm):
    task, work, out = fake_arm
    path = _arm(2, "kshot", work, out)
    record = json.loads(path.read_text())
    assert record["config"]["epochs"] == 0 and record["adapter"]["file"] is None
    assert record["validity"] == "ceiling"  # a perfect fake trips B.5 on the baseline arm
    assert not list(work.glob("*_adapter.pt"))
    path.unlink()
    journal = work / (rerun.ckpt_stem(2, "kshot") + "_docs.jsonl")
    journal.write_text(journal.read_text().splitlines()[0] + "\n")
    adapts = _FakePredictor.calls["adapt"]
    record = json.loads(_arm(2, "kshot", work, out).read_text())
    assert record["restarts"] == 1 and _FakePredictor.calls["adapt"] == adapts + 1


def test_an_empty_pool_is_journaled_as_its_own_reason(fake_arm):
    task, work, out = fake_arm
    _FakePredictor.empty_pool = {2}
    record = json.loads(_arm(3, "adapted", work, out).read_text())
    assert record["no_completion"] == 2
    assert record["no_completion_reasons"] == {rerun.OVER_CAP: 1, rerun.EMPTY_POOL: 1}
    assert record["predictions"][2]["reason"] == rerun.EMPTY_POOL
    assert rerun.excluded_indices(record, rerun.OVER_CAP) == [1]
    assert rerun.excluded_indices(record, rerun.EMPTY_POOL) == [2]
    assert rerun.excluded_indices(record) == [1, 2]


def test_journal_loader_drops_a_torn_last_line_keeps_first_duplicates_and_refuses_bad_indices(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"index": 0, "a": 1}\n{"index": 0, "a": 2}\n{"index": 1, "a":')
    assert rerun._load_journal(path) == [{"index": 0, "a": 1}]
    assert path.read_text() == '{"index": 0, "a": 1}\n'  # rewritten without the fragment
    path.write_text('{"index": 0, "a":\n{"index": 1, "a": 2}\n')
    with pytest.raises(json.JSONDecodeError):
        rerun._load_journal(path)
    path.write_text('{"index": 7, "a": 1}\n')
    with pytest.raises(SystemExit):
        rerun._load_journal(path, expected=4)
