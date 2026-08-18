"""Command-line adaptation: bring a task file, get ranked predictions.

This is the product seed: the same engine the competition kernel runs,
pointed at a user-supplied task. The task file uses the ARC schema
(train pairs with outputs; test pairs with inputs, outputs optional —
when present they are scored and reported).

    python -m arcttt.cli path/to/task.json --model <hf-model-dir> \
        --raw-format --dfs --rank 16 --output predictions.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from arcttt.augment import DIHEDRAL_SWEEP, expanded_sweep
from arcttt.model import CausalLMPredictor, TTTConfig
from arcttt.solve import SolveConfig, solve_task
from arcttt.tasks import grid_to_lists, load_task, score_attempts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="ARC-format task JSON file")
    parser.add_argument("--model", required=True, help="HF model directory or id")
    parser.add_argument("--output", help="where to write predictions JSON")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha (rslora)")
    parser.add_argument("--epochs", type=int, default=1, help="TTT epochs (0 = no TTT)")
    parser.add_argument("--batch-size", type=int, default=1, help="TTT examples per step")
    parser.add_argument("--palettes", type=int, default=0,
                        help="color permutations per dihedral element for TTT")
    parser.add_argument("--raw-format", action="store_true",
                        help="champion-style raw <|im_start|> framing")
    parser.add_argument("--dfs", action="store_true", help="constrained DFS decoding")
    parser.add_argument("--cutoff", type=float, default=0.1,
                        help="DFS cumulative probability cutoff")
    parser.add_argument("--time-budget", type=float, default=None,
                        help="DFS wall-clock budget per test input (seconds)")
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    return parser


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = build_parser().parse_args(argv)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    task = load_task(args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.model)  # type: ignore[no-untyped-call]
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(
        device  # type: ignore[arg-type]
    )

    config = TTTConfig(
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        epochs=args.epochs,
        max_sequence_tokens=args.max_seq,
        raw_qwen_format=args.raw_format,
        gradient_checkpointing=device.type == "cuda",
        use_dfs=args.dfs,
        dfs_probability_cutoff=args.cutoff,
        dfs_time_budget_seconds=args.time_budget,
        shuffle_examples=True,
        ttt_batch_size=args.batch_size,
    )
    ttt_augmentations = (
        expanded_sweep(seed=0, palettes_per_element=args.palettes)
        if args.palettes
        else DIHEDRAL_SWEEP
    )
    solve = SolveConfig(
        augmentations=DIHEDRAL_SWEEP,
        samples_per_augmentation=1,
        rescore_augmentations=DIHEDRAL_SWEEP,
        ttt_augmentations=ttt_augmentations,
    )

    started = time.monotonic()
    predictor = CausalLMPredictor(model, tokenizer, config, device)
    ranked = solve_task(task, predictor, solve)
    elapsed = time.monotonic() - started

    predictions: list[dict[str, object]] = []
    result: dict[str, object] = {
        "task_id": task.task_id,
        "seconds": round(elapsed, 1),
        "predictions": predictions,
    }
    solved = scored = 0
    for test_index, attempts in enumerate(ranked):
        entry: dict[str, object] = {
            "attempts": [grid_to_lists(grid) for grid in attempts[:2]],
        }
        solution = task.test[test_index].output
        if solution is not None:
            scored += 1
            hit = bool(attempts and score_attempts(list(attempts), solution))
            entry["solved"] = hit
            solved += int(hit)
        predictions.append(entry)
    if scored:
        result["solved"] = solved
        result["scored"] = scored

    payload = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
