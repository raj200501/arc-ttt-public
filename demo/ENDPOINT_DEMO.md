# Endpoint demo: bring examples → adapted endpoint (CORD receipts)

A 2-minute terminal demo proving the adaptation loop is running code, not
slides: an HTTP endpoint receives k example receipts (default k=10; OCR text + labelled
JSON), fine-tunes a LoRA adapter on them at request time (test-time
training), and answers held-out receipts with the changed weights — same
model, same prompt, only the weights changed.

The captured transcript of a real run lives in
`demo/endpoint_demo_transcript.txt` (genuine, unedited `tee` capture — a
reference if the demo cannot be run live).

**Transcript caveat.** The transcript is an unedited k=5 `tee` capture
whose closing citation misstates the config ("k=10 … same config") and
cites the seed-0 pair (field F1 0.661 → 0.788) without variance context —
that pair is the favorable draw of the 4-arm sweep, whose mean is −1.3 F1
(`experiments/cord_variance_summary_2026-08-08.json`). The citation is
fixed in `demo/endpoint_demo.py`; a k=10 re-capture is pending. The
transcript itself is left unedited on purpose.

## Reproduce

```bash
cd arc-ttt
bash demo/run_endpoint_demo.sh 2>&1 | tee demo/endpoint_demo_transcript.txt
```

That's the whole demo: the script starts the endpoint
(`demo/endpoint_demo.py serve`), waits for `/health`, runs the client
narrative (`demo/endpoint_demo.py demo`), and stops the endpoint.
Environment overrides: `PYTHON` (interpreter with the project deps),
`PORT`, `MODEL`, `DATA`, `K`, `EVAL_N`, `SEED`. Requirements: the project
venv (torch + transformers), the Qwen2.5-0.5B-Instruct weights in the HF
cache (~950 MB; downloaded automatically on first run), and
`demo/cord_validation.jsonl` (checked in; re-fetchable with
`python scripts/fetch_cord.py --split validation --limit 100 --out demo/cord_validation.jsonl`).

Everything is deterministic (fixed seed, greedy decoding), so a re-run
reproduces the same model outputs; only the timings vary.

## What the viewer is seeing

1. `GET /health` — the endpoint is a live HTTP server holding the model.
2. A deterministic split of 100 real CORD receipts: k (default 10)
   become the "customer's examples", 2 are held out as graded questions.
3. **STEP A (before)** — `POST /adapt {"adapt": false}`: the k examples go
   in the prompt only (standard few-shot prompting, zero weight updates —
   the injected LoRA is zero-initialised, a true no-op). The model answers
   the 2 held-out receipts; each answer is printed side-by-side with the
   human gold labels, with field-level micro-F1.
4. **STEP B (after)** — `POST /adapt {"adapt": true}`: same payload, but
   now the endpoint trains a LoRA adapter on the same k examples before
   answering the same 2 receipts.
5. **Scoreboard** — before vs after, per receipt: fields correct and
   field F1. The only difference between the arms is the weight update.

## Timing (from the captured k=5 run, single 4-core CPU)

| stage | seconds |
|---|---|
| model load (from local HF cache) | 85.4 |
| STEP A round-trip (2 queries, no adaptation) | 78.2 (answers 47.7 + 29.5) |
| adaptation (LoRA on 5 examples, 1 epoch) | 157.8 |
| STEP B round-trip (adapt + 2 queries) | 189.3 (answers 17.0 + 14.4) |
| total demo wall time | 377.5 |

## Honest caveats

- **Dev-scale model.** Qwen2.5-0.5B-Instruct, CPU, float32 — chosen so the
  loop runs live in minutes. The mechanism, not the model, is the product.
- **One schema, one dataset.** CORD-v2 receipts (CC BY 4.0), text-only
  post-OCR, gt_parse restricted to the released superclasses — the same
  setup as the measured smoke.
- **n=2 is a demo, not an eval — and the demo proves the mechanism, not
  quality.** What this demo shows is that the weights change at request
  time, in minutes, on a CPU. The measured quality picture lives in the
  registry's cord-variance row: across 4 paired seed/k arms (k=10 seeds
  0/2/3, k=5 seed 1; 20 held-out receipts each; 0.5B dev model, 1-epoch
  TTT), adaptation is net-neutral — mean −1.3 F1, range −6.5 to +12.7,
  only the original seed-0 arm positive; valid JSON 19–20/20 in every
  arm. Per the preregistered spec
  (`docs/research/ENTERPRISE_EVAL_SPEC.md` §4), the quality claim at
  dev scale is an open empirical question; the named next lever is
  model scale. The k=30 arm's adapted side never completed — it was
  interrupted three times by container restarts and is reported as-is;
  its k-shot baseline did land at 0.672, the strongest prompted arm
  (`experiments/cord_k30_seed1_kshot_2026-08-08.json`). Artifacts:
  `experiments/cord_variance_summary_2026-08-08.json` + per-arm files.
- **Endpoint plumbing.** The HTTP layer is the shipped server machinery
  (`arcttt.serve.make_handler` + stdlib `HTTPServer`) driving the shipped
  text-mode TTT engine (`arcttt.text_ttt.TextPredictor`). The shipped
  `/adapt` payload parser in `src/arcttt/serve.py` currently accepts only
  ARC grid tasks (`task_from_payload` → `to_grid`, serve.py:29-52), so
  this demo supplies a text-mode payload parser/service in
  `demo/endpoint_demo.py` (new file; `src/` untouched). Folding text-mode
  payloads into the shipped server is a small, known follow-up for the
  serving lane.
- Per-request adaptation is stateless by design (each POST trains a fresh
  adapter); persisting adapters per customer is product roadmap, not in
  this demo.
