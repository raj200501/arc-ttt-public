#!/usr/bin/env python3
"""E-r2 tokenizer seed screen — the frozen length-only, outcome-blind gate.

Reproduces (and makes rerunnable/bankable) the screen E-r2 froze: for
each candidate seed, build the exact diverse-compact k=30 corpus and
measure, with the production tokenizer and chat template, (a) the max
LOO training-sequence token length (text_ttt_training_examples, the
same sequences adapt_text trains on) and (b) the max decode-prompt
token length (text_task_to_messages with demos). A seed is ELIGIBLE
iff both maxima are <= the frozen 7,900 screen bound (headroom under
the 8,192 budget); the screen reads NO outcomes — lengths only.

    PYTHONPATH=src python3 scripts/screen_er_seeds.py 201 202 ... [--out FILE]
"""

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

BOUND = 7900  # frozen E-r2 screen bound
BUDGET = 8192  # frozen B max_sequence_tokens at k=30


def main(argv: list[str]) -> int:
    from transformers import AutoTokenizer
    from arcttt.novel_schema import make_task
    from arcttt.text_ttt import (text_task_to_messages,
                                 text_ttt_training_examples)

    out = None
    if "--out" in argv:
        i = argv.index("--out")
        out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    seeds = [int(a) for a in argv]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    def n_tokens(turns) -> int:
        msgs = [{"role": t.role, "content": t.content} for t in turns]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    rows = []
    for seed in seeds:
        task, schema = make_task(seed=seed, n_train=30, n_test=60,
                                 task_id=f"er-screen-{seed}",
                                 geometry="diverse-compact")
        loo_max = max(n_tokens(ex) for ex in text_ttt_training_examples(task))
        dec = [n_tokens(text_task_to_messages(task, i))
               for i in range(len(task.test))]
        row = {"seed": seed, "tenant": schema.tenant_id,
               "max_loo_train_tokens": loo_max,
               "decode_prompt_tokens_min": min(dec),
               "decode_prompt_tokens_max": max(dec),
               "eligible": loo_max <= BOUND and max(dec) <= BOUND}
        rows.append(row)
        print(row, flush=True)

    if out:
        artifact = {
            "artifact": "E-r2 tokenizer seed screen (length-only, "
                        "outcome-blind; spec E-r2)",
            "geometry": "diverse-compact", "k": 30, "eval_n": 60,
            "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct (chat template)",
            "screen_bound": BOUND, "sequence_budget": BUDGET,
            "seeds": rows,
        }
        pathlib.Path(out).write_text(json.dumps(artifact, indent=1))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
