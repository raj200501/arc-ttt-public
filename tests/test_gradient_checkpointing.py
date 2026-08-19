"""Regression tests for the gradient-checkpointing edges of adapt/eval.

Two edges, both claimed in the paper (section 3.5) and both cheap to break
silently:

- ``enable_input_require_grads``: with only LoRA params trainable, the
  frozen embeddings give the checkpointed segment no input requiring grad;
  without the hook, backward under checkpointing fails (or trains nothing).
- the adapt/eval transition must disable checkpointing and restore the
  original (unwrapped) layer forwards before generation — on success AND
  on an exception mid-training (the CUDA OOM this feature exists for),
  so an OOM-ladder retry re-wraps clean original forwards.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from test_model_loop import make_task, tiny_model, tiny_tokenizer  # noqa: E402

from arcttt.augment import IDENTITY  # noqa: E402
from arcttt.model import CausalLMPredictor, TTTConfig  # noqa: E402


def test_adapt_with_gradient_checkpointing_trains_and_disables_it_again() -> None:
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    predictor = CausalLMPredictor(
        model,
        tokenizer,
        TTTConfig(lora_rank=2, lora_alpha=4, epochs=1, max_new_tokens=8,
                  gradient_checkpointing=True),
        torch.device("cpu"),
    )

    predictor.adapt(make_task(), (IDENTITY,))

    # (a) backward under checkpointing succeeded: without
    # enable_input_require_grads() the LoRA-only setup has no checkpointed
    # input requiring grad and the optimizer step moves nothing.
    lora_b = [
        parameter
        for name, parameter in predictor.model.named_parameters()
        if "lora_b" in name
    ]
    assert lora_b and any(w.abs().sum().item() > 0 for w in lora_b), (
        "gradient-checkpointed TTT must actually update LoRA weights"
    )
    # (b) the adapt/eval transition disabled checkpointing before generation
    # and left the model in eval mode.
    assert not model.is_gradient_checkpointing
    assert not model.training


def test_exception_mid_training_still_restores_forwards_and_eval_mode() -> None:
    # chunked_loss_tokens > 0 + gradient_checkpointing engages the manual
    # torch.utils.checkpoint layer wrapping; the try/finally must unwind the
    # wrappers, disable checkpointing, and return to eval mode even when
    # training raises (the OOM path).
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    predictor = CausalLMPredictor(
        model,
        tokenizer,
        TTTConfig(lora_rank=2, lora_alpha=4, epochs=1, max_new_tokens=8,
                  gradient_checkpointing=True, chunked_loss_tokens=4),
        torch.device("cpu"),
    )

    def boom(*args, **kwargs):
        raise RuntimeError("simulated mid-training OOM")

    predictor._chunked_loss_backward = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated mid-training OOM"):
        predictor.adapt(make_task(), (IDENTITY,))

    # wrappers unwound: no layer forward is the checkpoint closure
    for layer in model.model.layers:
        assert "wrapped" not in layer.forward.__qualname__
    assert not model.is_gradient_checkpointing
    assert not model.training

    # and a subsequent SUCCESSFUL adapt on the same model works cleanly
    del predictor._chunked_loss_backward  # restore the real method
    predictor.adapt(make_task(), (IDENTITY,))
    for layer in model.model.layers:
        assert "wrapped" not in layer.forward.__qualname__
    assert not model.is_gradient_checkpointing
    assert not model.training


def test_all_examples_dropped_is_reported_loudly(capsys) -> None:
    # Companion to the exception path: an all-dropped adaptation must not be
    # silent (zero optimizer steps leaves a no-op adapter injected).
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    predictor = CausalLMPredictor(
        model,
        tokenizer,
        TTTConfig(lora_rank=2, lora_alpha=4, epochs=1, max_new_tokens=8,
                  max_sequence_tokens=3),  # every example exceeds this
        torch.device("cpu"),
    )
    predictor.adapt(make_task(), (IDENTITY,))
    out = capsys.readouterr().out
    assert "dropped" in out
    assert "ALL examples dropped" in out
    assert not model.training
