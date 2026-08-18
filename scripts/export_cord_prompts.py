"""Export the exact k-shot CORD prompts the smoke harness would build.

Reproduces enterprise_smoke.py's split (same seed shuffle, k demos,
eval_n held-out receipts) and serializes each eval receipt's chat turns
(demo user/assistant pairs + test user turn) plus the gold output to a
JSONL work file, so an external model can be scored on the identical
task with arcttt.text_ttt.score_text_output.

    python scripts/export_cord_prompts.py --data demo/cord_validation.jsonl \
        --k 10 --eval-n 20 --seed 0 --out <work>.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arcttt.text_task import from_cord_gt
from arcttt.text_ttt import text_task_to_messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eval-n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if len(rows) < args.k + args.eval_n:
        raise SystemExit(f"need >= {args.k + args.eval_n} rows, have {len(rows)}")
    task = from_cord_gt(
        rows[: args.k],
        rows[args.k : args.k + args.eval_n],
        task_id=f"cord-k{args.k}-seed{args.seed}",
    )

    with Path(args.out).open("w") as handle:
        for index in range(len(task.test)):
            gold = task.test[index].output_text
            assert gold is not None
            turns = text_task_to_messages(task, index)
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "messages": [
                            {"role": turn.role, "content": turn.content}
                            for turn in turns
                        ],
                        "gold": gold,
                    }
                )
                + "\n"
            )
    print(f"wrote {len(task.test)} prompts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
