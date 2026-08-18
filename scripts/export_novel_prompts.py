"""Export the exact k-shot novel-schema prompts the gate kernel builds.

Mirrors export_cord_prompts.py but for the Addendum B corpus: the task is
constructed with the SAME make_task call as kaggle/entry_novel_schema.py
(n_train=k, n_test=60), so exported prompts are the identical documents
the kernel arms saw. Prompts and gold go to SEPARATE files so an external
model (including an agent-harness frontier arm) can predict without the
answers in view; scoring joins them by index afterwards.

    python scripts/export_novel_prompts.py --seed 1 --k 10 --limit 20 \
        --prompts-out work/novel_s1_prompts.jsonl \
        --gold-out work/novel_s1_gold.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arcttt.novel_schema import make_task
from arcttt.text_ttt import text_task_to_messages

EVAL_N = 60  # frozen Addendum B value; --limit slices, never re-generates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=EVAL_N,
                        help="export only the first N eval docs (subset is "
                        "index-labeled so partial arms are honest)")
    parser.add_argument("--prompts-out", required=True)
    parser.add_argument("--gold-out", required=True)
    args = parser.parse_args(argv)

    task, schema = make_task(
        seed=args.seed,
        n_train=args.k,
        n_test=EVAL_N,
        task_id=f"novel-0.5b-k{args.k}-seed{args.seed}",
    )
    n = min(args.limit, len(task.test))
    with Path(args.prompts_out).open("w") as ph, Path(args.gold_out).open("w") as gh:
        for index in range(n):
            gold = task.test[index].output_text
            assert gold is not None
            turns = text_task_to_messages(task, index)
            ph.write(json.dumps({
                "index": index,
                "seed": args.seed,
                "k": args.k,
                "messages": [{"role": t.role, "content": t.content} for t in turns],
            }) + "\n")
            gh.write(json.dumps({"index": index, "seed": args.seed, "gold": gold}) + "\n")
    print(f"tenant {schema.tenant_id}: wrote {n}/{len(task.test)} prompts "
          f"to {args.prompts_out}, gold to {args.gold_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
