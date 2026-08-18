"""The chunked loss must be the HF labels-path loss, gradients included.

``chunked_loss_tokens`` exists purely as a memory optimization; if it shifts
the training math even slightly, the Addendum B k=30 arms stop being the
frozen protocol and the gate result is invalid. So this pins, on a fake
causal LM that reproduces HF's exact loss semantics (shift by one, fp32
cross-entropy, mean over non-ignored tokens):

- the accumulated LoRA-parameter gradients, chunked vs labels-path
- across chunk sizes that do and do not divide the sequence length
- with ignore_index masking present (the prompt tokens)

If any of these drift, the kernel change is NOT math-preserving and may not
ship.
"""

from __future__ import annotations

import torch

from arcttt.model import CausalLMPredictor, TTTConfig


class _FakeTrunkOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _FakeTrunk(torch.nn.Module):
    """Stands in for the HF decoder: embeddings + one trainable projection."""

    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.proj = torch.nn.Linear(hidden, hidden)  # plays the LoRA role

    def forward(self, input_ids=None, attention_mask=None):
        h = self.proj(self.embed(input_ids))
        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1)
        return _FakeTrunkOutput(h)


class _FakeLMOutput:
    def __init__(self, loss: torch.Tensor) -> None:
        self.loss = loss


class _FakeCausalLM(torch.nn.Module):
    """Mimics HF CausalLM: .model trunk, .lm_head, and labels-path loss.

    The labels-path forward reproduces transformers' semantics exactly:
    logits shifted against labels by one, cross-entropy computed on
    ``logits.float()``, mean over tokens whose label is not -100.
    """

    def __init__(self, vocab: int = 61, hidden: int = 16) -> None:
        super().__init__()
        self.model = _FakeTrunk(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight.requires_grad_(False)  # frozen, as under LoRA

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        hidden = self.model(input_ids, attention_mask).last_hidden_state
        logits = self.lm_head(hidden)
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )
        return _FakeLMOutput(loss)


def _grads_for(chunked_tokens: int, seq_len: int = 23) -> list[torch.Tensor]:
    torch.manual_seed(7)
    lm = _FakeCausalLM()
    # Trainable params = the trunk projection (the "LoRA" stand-in).
    for p in lm.model.embed.parameters():
        p.requires_grad_(False)

    shell = CausalLMPredictor.__new__(CausalLMPredictor)  # no HF download; only the method
    shell.model = lm
    shell.config = TTTConfig(chunked_loss_tokens=chunked_tokens)

    torch.manual_seed(11)
    input_ids = torch.randint(0, 61, (1, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[0, : seq_len // 2] = -100  # prompt tokens are masked, as in TTT

    lm.zero_grad(set_to_none=True)
    if chunked_tokens > 0:
        shell._chunked_loss_backward(input_ids, attention_mask, labels)
    else:
        lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss.backward()
    return [p.grad.clone() for p in lm.model.proj.parameters()]


def test_chunked_gradients_equal_labels_path_gradients() -> None:
    reference = _grads_for(0)
    for chunk in (4, 5, 8, 64):  # divides seq-1, doesn't, and exceeds it
        chunked = _grads_for(chunk)
        for ref, got in zip(reference, chunked):
            assert torch.allclose(ref, got, atol=1e-6), (
                f"gradient drift at chunk={chunk}: max "
                f"{(ref - got).abs().max().item()}"
            )


def test_all_labels_masked_is_a_clean_noop() -> None:
    torch.manual_seed(3)
    lm = _FakeCausalLM()
    shell = CausalLMPredictor.__new__(CausalLMPredictor)
    shell.model = lm
    shell.config = TTTConfig(chunked_loss_tokens=8)
    ids = torch.randint(0, 61, (1, 12))
    labels = torch.full_like(ids, -100)
    lm.zero_grad(set_to_none=True)
    shell._chunked_loss_backward(ids, torch.ones_like(ids), labels)
    assert all(p.grad is None for p in lm.model.proj.parameters())
