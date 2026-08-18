"""Pure-torch LoRA injection tests (no peft dependency)."""

from __future__ import annotations

import torch
from torch import nn

from arcttt.lora import (
    LoRALinear,
    inject_lora,
    lora_parameters,
    remove_lora,
    trainable_parameter_count,
)


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.block = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
        self.lm_head = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.block(self.a(x)))


def test_inject_wraps_linears_and_freezes_base() -> None:
    model = Tiny()
    wrapped = inject_lora(model, rank=2, alpha=4)
    assert "a" in wrapped and any("block" in w for w in wrapped)
    assert "lm_head" not in wrapped  # skipped by default
    # only lora_ params train
    for name, p in model.named_parameters():
        if "lora_" in name:
            assert p.requires_grad
        else:
            assert not p.requires_grad
    assert trainable_parameter_count(model) == sum(p.numel() for p in lora_parameters(model))


def test_lora_starts_as_identity_then_learns() -> None:
    torch.manual_seed(0)
    model = Tiny()
    x = torch.randn(4, 8)
    base_out = model(x).clone()
    inject_lora(model, rank=4, alpha=8)
    # B initialized to zero => delta is zero => output unchanged at injection
    assert torch.allclose(model(x), base_out, atol=1e-5)

    target = torch.randn(4, 4)
    opt = torch.optim.SGD(lora_parameters(model), lr=0.1)
    first = None
    for _ in range(50):
        opt.zero_grad()
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
    assert float(loss) < first  # LoRA params actually reduce the loss


def test_remove_restores_base_and_is_idempotent() -> None:
    model = Tiny()
    inject_lora(model, rank=2, alpha=4)
    assert any(isinstance(m, LoRALinear) for m in model.modules())
    remove_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    remove_lora(model)  # no-op second time
    assert isinstance(model.a, nn.Linear)


def test_reinjection_after_removal_works() -> None:
    model = Tiny()
    inject_lora(model, rank=2, alpha=4)
    remove_lora(model)
    wrapped = inject_lora(model, rank=3, alpha=6)  # fresh adapter, different rank
    assert wrapped
    assert model.a.rank == 3
