#!/usr/bin/env python3
"""Challenger-side kit for the blind-holdout protocol.

Runs on the CHALLENGER's machine so gold labels never leave it.
Two subcommands:

  split  — stdlib-only. Takes one JSONL of labeled documents
           ({"id": ..., "text": ..., "gold": {...}} per line), makes a
           deterministic seeded split, and writes a challenge package:
             train.jsonl        (id, text, gold)  -> send to the founder
             holdout.jsonl      (id, text)        -> send to the founder
             gold_holdout.jsonl (id, gold)        -> KEEP PRIVATE
             TERMS.md           (filled skeleton) -> send + countersign
           Prints the gold file's sha256 and the optional OpenTimestamps
           stamp command so the challenger can anchor their gold before
           anything ships (protocol item 8).

  score  — one-time scoring pass with the pinned scorer
           (arcttt.text_ttt.score_text_output at the commit named in the
           terms; needs `pip install torch` + `pip install -e .`).
           Aggregation per the protocol: mean per-document micro-F1,
           invalid or missing JSON scores 0, every holdout id counted
           exactly once.

The founder never runs `split` or `score` on a real challenge — that is
the point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys

TERMS_SKELETON = """# Blind-holdout challenge terms — {name}

Protocol: BLIND_HOLDOUT_PROTOCOL.md as anchored in the newest snapshot
in docs/research/ (the snapshot governs). Scorer: `score_text_output`
at repository commit {commit}; aggregation is mean
per-document micro-F1 with invalid JSON scored 0.

## Target schema (challenger: declare before sending)

⟨FIELD NAMES + TYPES, and every normalization convention your gold
relies on (date formats, digits-only numbers, casing) — non-verbatim
gold is only fair if its conventions are declared here.⟩

- Documents: {n_total} total — {n_train} labeled training pairs
  (train.jsonl), {n_holdout} held out (holdout.jsonl; gold withheld).
- Split: deterministic, seed {seed}, by this kit
  (scripts/make_challenge.py); gold_holdout.jsonl sha256 {gold_sha}.
- Base model: name + immutable revision + checkpoint sha256 are
  required deliverables (regenerable, not just hashed).
- Submission: predictions.jsonl, one line per holdout id
  ({{"id": ..., "prediction": {{...}}}}), each id exactly once;
  missing or duplicate ids score 0 for that document. Single
  submission; 72h from document receipt.
- Deliverables with the submission: adapter weights + sha256, exact
  repo commit, exact adaptation command, seed.
- Result publishes either way, terms attached.
"""


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_labeled_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    seen_ids: set = set()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"line {i}: not valid JSON ({e})")
            for key in ("id", "text", "gold"):
                if key not in row:
                    sys.exit(f"line {i}: missing required field '{key}'")
            if not isinstance(row["gold"], dict):
                sys.exit(f"line {i}: 'gold' must be a JSON object")
            if row["id"] in seen_ids:
                sys.exit(f"line {i}: duplicate id {row['id']!r}")
            seen_ids.add(row["id"])
            rows.append(row)
    if len(rows) < 3:
        sys.exit(f"need at least 3 labeled documents, got {len(rows)}")
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_split(args: argparse.Namespace) -> int:
    rows = read_labeled_jsonl(pathlib.Path(args.docs))
    if not 0 < args.train_k < len(rows):
        sys.exit(f"--train-k must be in [1, {len(rows) - 1}] for {len(rows)} docs")
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    train = [rows[i] for i in sorted(order[: args.train_k])]
    holdout = [rows[i] for i in sorted(order[args.train_k:])]

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl",
                [{"id": r["id"], "text": r["text"], "gold": r["gold"]} for r in train])
    write_jsonl(out / "holdout.jsonl",
                [{"id": r["id"], "text": r["text"]} for r in holdout])
    gold_path = out / "gold_holdout.jsonl"
    write_jsonl(gold_path, [{"id": r["id"], "gold": r["gold"]} for r in holdout])
    gold_sha = sha256_file(gold_path)
    (out / "TERMS.md").write_text(TERMS_SKELETON.format(
        name=args.name, n_total=len(rows), n_train=len(train),
        n_holdout=len(holdout), seed=args.seed, gold_sha=gold_sha,
        commit=args.commit or "⟨PIN COMMIT HERE before sending⟩"))

    print(f"challenge package written to {out}/")
    print(f"  SEND:   train.jsonl ({len(train)} labeled), "
          f"holdout.jsonl ({len(holdout)} unlabeled), TERMS.md")
    print(f"  KEEP:   gold_holdout.jsonl  sha256 {gold_sha}")
    print("  anchor your gold before sending anything (protocol item 8):")
    print(f"    ots stamp {gold_path}   (pip install opentimestamps-client)")
    if not args.commit:
        print("  BEFORE SENDING: pin the scorer commit in TERMS.md "
              "(or re-run with --commit <sha>) and fill the schema section.")
    print("  fill the '## Target schema' section in TERMS.md with your "
          "field names and normalization conventions.")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    # Runnable as a bare script from a clone: put the repo's src/ on the
    # path before importing the pinned scorer.
    repo_src = pathlib.Path(__file__).resolve().parents[1] / "src"
    if repo_src.is_dir() and str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from arcttt.text_ttt import score_text_output  # the pinned scorer

    gold = {}
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                gold[row["id"]] = row["gold"]

    preds: dict = {}
    duplicate: set = set()
    with open(args.pred, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["id"] in preds:
                    duplicate.add(row["id"])
                preds[row["id"]] = row.get("prediction")

    per_doc = []
    invalid = 0
    for doc_id, gold_obj in sorted(gold.items(), key=lambda kv: str(kv[0])):
        pred = preds.get(doc_id)
        if doc_id in duplicate or pred is None:
            per_doc.append((doc_id, 0.0))
            invalid += 1
            continue
        score = score_text_output(json.dumps(pred), json.dumps(gold_obj))
        f1 = score.micro_f1 if score.valid_json else 0.0
        invalid += 0 if score.valid_json else 1
        per_doc.append((doc_id, f1))

    extra = sorted(set(preds) - set(gold))
    mean_f1 = sum(f1 for _, f1 in per_doc) / len(per_doc)
    for doc_id, f1 in per_doc:
        print(f"  {doc_id}: {f1:.4f}")
    if extra:
        print(f"  WARNING: {len(extra)} prediction ids not in holdout (ignored): {extra[:5]}")
    print(f"mean per-document micro-F1 over {len(per_doc)} holdout docs: "
          f"{mean_f1:.4f} ({invalid} scored 0 as invalid/missing/duplicate)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_split = sub.add_parser("split", help="build a challenge package (stdlib-only)")
    p_split.add_argument("--docs", required=True,
                         help="JSONL of labeled documents: {id, text, gold} per line")
    p_split.add_argument("--train-k", type=int, default=20,
                         help="labeled training pairs to release (default 20)")
    p_split.add_argument("--seed", type=int, default=0, help="split seed (default 0)")
    p_split.add_argument("--out-dir", default="challenge", help="output directory")
    p_split.add_argument("--name", default="unnamed-challenge", help="challenge name for TERMS.md")
    p_split.add_argument("--commit", default=None,
                         help="repository commit to pin the scorer to in TERMS.md")
    p_split.set_defaults(func=cmd_split)

    p_score = sub.add_parser("score", help="one-time scoring pass with the pinned scorer")
    p_score.add_argument("--pred", required=True, help="founder's predictions.jsonl")
    p_score.add_argument("--gold", required=True, help="your private gold_holdout.jsonl")
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
