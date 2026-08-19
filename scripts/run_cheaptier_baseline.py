#!/usr/bin/env python3
"""M2: cheap-API-tier k-shot baseline on the novel-schema corpus.

Answers the question the 2026-08-19 cost analysis left open: does the
CHEAPEST current API tier hold quality on never-seen schemas the way
the frontier tier does (frontier k-shot = 1.00,
experiments/novel_frontier_baseline_2026-08-16.json)? Either answer
publishes: 1.00 narrows our cost wedge and we say so; a miss means the
tier that actually competes with us on price does not match the
adapted 0.5B's banked quality.

Protocol parity with the frontier baseline artifact: same
make_task(seed, n_train=10, n_test=60) corpus, same first-20 eval docs
per tenant (indices 0-19, bounded subset labeled as such), single
sample per receipt, temperature 0, scored with
arcttt.text_ttt.score_text_output. Prompt = the exact
text_task_to_messages turns the kernel arms build, mapped to the API's
chat roles. Context arm, NOT a gate arm — never pooled with kernel
arms.

The API key is read from a file OUTSIDE the repository and never
stored in the artifact.

Usage:
    PYTHONPATH=src python3 scripts/run_cheaptier_baseline.py \
        <key_file> <model_id> <out.json>
"""

import json
import os
import pathlib
import sys
import time
import urllib.request


def call_gemini(key: str, model: str, turns) -> tuple[str, dict]:
    contents = []
    for t in turns:
        role = "model" if t.role == "assistant" else "user"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append({"text": t.content})
        else:
            contents.append({"role": role, "parts": [{"text": t.content}]})
    body = json.dumps({
        "contents": contents,
        "generationConfig": {"temperature": 0},
    }).encode()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    req = urllib.request.Request(url, data=body, headers={
        "x-goog-api-key": key, "Content-Type": "application/json"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            parts = data["candidates"][0]["content"].get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            return text, data.get("usageMetadata", {})
        except Exception as exc:  # 429/5xx/network — retry with backoff
            if attempt == 7:
                raise
            wait = min(2 ** (attempt + 1), 120)
            print(f"retry in {wait}s: {exc}", flush=True)
            time.sleep(wait)


def main() -> int:
    key_file, model, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    key = open(key_file).read().strip()
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
    from arcttt.novel_schema import make_task
    from arcttt.text_ttt import text_task_to_messages, score_text_output

    journal = out_path + ".journal.jsonl"
    rows = []
    if os.path.exists(journal):
        rows = [json.loads(l) for l in open(journal) if l.strip()]
    done = {(r["seed"], r["index"]) for r in rows}

    usage_totals = {"prompt": 0, "output": 0}
    for seed in (1, 2, 3):
        task, schema = make_task(seed=seed, n_train=10, n_test=60,
                                 task_id=f"novel-0.5b-k10-seed{seed}")
        for index in range(20):
            if (seed, index) in done:
                continue
            turns = text_task_to_messages(task, index)
            time.sleep(6)  # free-tier RPM pacing
            text, usage = call_gemini(key, model, turns)
            gold = task.test[index].output_text
            score = score_text_output(text, gold)
            row = {"seed": seed, "tenant": schema.tenant_id, "index": index,
                   "prediction": text, "micro_f1": round(score.micro_f1, 4),
                   "exact": score.micro_f1 == 1.0,
                   "prompt_tokens": usage.get("promptTokenCount", 0),
                   "output_tokens": usage.get("candidatesTokenCount", 0)}
            rows.append(row)
            with open(journal, "a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            print(f"seed {seed} doc {index}: f1={row['micro_f1']}", flush=True)

    per_tenant = []
    for seed in (1, 2, 3):
        srows = [r for r in rows if r["seed"] == seed]
        per_tenant.append({
            "seed": seed, "tenant": srows[0]["tenant"], "n": len(srows),
            "mean_micro_f1": round(sum(r["micro_f1"] for r in srows) / len(srows), 4),
            "exact": sum(1 for r in srows if r["exact"])})
    usage_totals["prompt"] = sum(r["prompt_tokens"] for r in rows)
    usage_totals["output"] = sum(r["output_tokens"] for r in rows)

    artifact = {
        "artifact": "cheap-API-tier k-shot baseline, novel-schema corpus "
                    "(context arm, NOT a gate arm)",
        "date": "2026-08-19",
        "model": model + " (cheapest current API tier, k-shot in-context, "
                 "no adaptation; self-run and labeled as such)",
        "spec_relation": "Same corpus construction as the frontier baseline "
                         "(make_task(seed, n_train=10, n_test=60)); scored on "
                         "the first 20 eval docs per tenant (indices 0-19). "
                         "Bounded subset — labeled as such, never pooled "
                         "with the 60-doc kernel arms.",
        "protocol": "Exact text_task_to_messages turns mapped to the API's "
                    "chat roles; temperature 0; single sample per receipt; "
                    "scored with arcttt.text_ttt.score_text_output (same "
                    "scorer as all arms). Raw predictions stored.",
        "honest_framing_REQUIRED": "Context arm for the cost analysis "
                    "(COST_APPENDIX 2026-08-19): whether the cheap tier "
                    "matches frontier quality on never-seen schemas. States "
                    "nothing about any preregistered gate.",
        "k": 10,
        "pooled_mean_micro_f1": round(sum(r["micro_f1"] for r in rows) / len(rows), 4),
        "pooled_n": len(rows),
        "per_tenant": per_tenant,
        "api_usage_tokens": usage_totals,
        "per_doc": rows,
    }
    tmp = out_path + ".tmp"
    open(tmp, "w").write(json.dumps(artifact, indent=1))
    os.replace(tmp, out_path)
    print(json.dumps({"pooled": artifact["pooled_mean_micro_f1"],
                      "per_tenant": per_tenant}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
