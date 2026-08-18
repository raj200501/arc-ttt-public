"""CPU smoke for the micro-train kernel bundle: ~20 steps on 2 tiny tasks.

Runs against kaggle/micro/bundled_pipeline.py (the exact file Kaggle will
execute), not the source tree: a tiny Qwen2 with a raw-qwen-capable toy
tokenizer trains for 20 optimizer steps on two 2x2 tasks, and the gates are
(1) the loop executes end to end, (2) the loss moves down, (3) the atomic
adapter checkpoint + loss-curve JSON are written and readable, (4) the
paired-eval path (FrozenPredictor -> solve_task -> lp_true) executes.

Usage: python kaggle/micro/smoke_micro_train.py [workdir]
"""

from __future__ import annotations

import importlib.util
import json
import string
import sys
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

BUNDLE = Path(__file__).resolve().parent / "bundled_pipeline.py"


def load_bundle():
    spec = importlib.util.spec_from_file_location("bundled_pipeline", BUNDLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves cls.__module__ here
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Char-level toy tokenizer that supports the raw <|im_start|> framing."""

    vocab = {ch: i for i, ch in enumerate("0123456789")}
    vocab["\n"] = len(vocab)
    for ch in string.ascii_lowercase:  # covers the "user"/"assistant" role words
        vocab[ch] = len(vocab)
    for special in ("<|im_start|>", "<|im_end|>", "<|pad|>"):
        vocab[special] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<|pad|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<|pad|>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )


def tiny_model(vocab_size: int) -> Qwen2ForCausalLM:
    torch.manual_seed(11)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    return Qwen2ForCausalLM(config)


def main() -> int:
    workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()

    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    grid_a = ((1, 2), (3, 4))
    grid_b = ((5, 6), (7, 8))
    tasks = [
        bundle.Task(
            task_id=f"tiny{i}",
            train=(
                bundle.Pair(input=g, output=g),
                bundle.Pair(input=g, output=g),
            ),
            test=(bundle.Pair(input=g, output=g),),
        )
        for i, g in enumerate((grid_a, grid_b))
    ]

    adapter_path = str(workdir / "adapter_micro.safetensors")
    log_path = str(workdir / "train_log.json")
    log = bundle.train_micro(
        model,
        tokenizer,
        tasks,
        torch.device("cpu"),
        deadline=time.time() + 3600,
        adapter_path=adapter_path,
        log_path=log_path,
        max_steps=20,
        grad_accum=1,
        checkpoint_every=7,
        learning_rate=5e-3,
        warmup_steps=2,
        lora_rank=4,
        lora_alpha=8,
        max_sequence_tokens=256,
        seed=0,
    )

    steps = log["steps"]
    assert len(steps) == 20, f"expected 20 optimizer steps, got {len(steps)}"
    first, last = steps[0]["loss"], steps[-1]["loss"]
    assert last < first, f"loss did not move down: {first:.4f} -> {last:.4f}"

    tensors = load_file(adapter_path)
    assert tensors and all("lora_" in name for name in tensors)
    disk_log = json.loads(Path(log_path).read_text())
    assert disk_log["optimizer_steps"] == 20 and len(disk_log["steps"]) == 20

    # Paired-eval path on the adapter-carrying model, smoke-sized budgets.
    bundle.eval_configs = lambda: (
        bundle.TTTConfig(
            epochs=0, max_new_tokens=8, max_sequence_tokens=256,
            raw_qwen_format=True, use_dfs=True, dfs_probability_cutoff=0.1,
            dfs_max_candidates=4, dfs_time_budget_seconds=2.0,
        ),
        bundle.SolveConfig(
            augmentations=bundle.AUG_QUARTET[:2],
            samples_per_augmentation=1,
            rescore_augmentations=bundle.AUG_QUARTET[:2],
        ),
    )
    report = bundle.run_eval(
        model, tokenizer, torch.device("cpu"), tasks[:1], "smoke"
    )
    entry = report["tasks"]["tiny0"]
    assert "error" not in entry, f"eval path failed: {entry}"
    assert report["scored_pairs"] == 1

    print(
        f"SMOKE OK | 20 steps | loss {first:.4f} -> {last:.4f} | "
        f"{len(tensors)} adapter tensors | tokens {log['tokens']} | "
        f"eval scored {report['scored_pairs']} pair(s), "
        f"solved {report['solved_pairs']}, mean lp_true {report['mean_lp_true']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
