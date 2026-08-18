"""Minimal LoRA for per-task adaptation, dependency-free beyond torch.

The Kaggle competition image does not ship ``peft`` and submission reruns are
offline, so per-task adaptation must not depend on it. This module injects
low-rank trainable deltas into the linear layers of a frozen model and can
remove them again, which is all the TTT loop needs. It is a small, tested
reimplementation of the standard LoRA update (rank-decomposed A@B added to a
frozen weight), not a general adapter framework.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """Wraps a frozen linear layer with a trainable low-rank delta."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, use_rslora: bool) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = rank
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        # scaling: rslora uses alpha/sqrt(rank), classic uses alpha/rank
        self.scaling = alpha / (math.sqrt(rank) if use_rslora else rank)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.base(x)
        delta = torch.nn.functional.linear(x, self.lora_a)
        delta = torch.nn.functional.linear(delta, self.lora_b)
        return out + self.scaling * delta


def _named_linears(module: nn.Module, prefix: str = "") -> list[tuple[str, nn.Linear]]:
    found = []
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            found.append((path, child))
        else:
            found.extend(_named_linears(child, path))
    return found


def _set_submodule(root: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    *,
    use_rslora: bool = True,
    skip_names: tuple[str, ...] = ("lm_head",),
) -> list[str]:
    """Replace every eligible ``nn.Linear`` with a LoRA-wrapped version.

    Returns the paths that were wrapped. All non-LoRA parameters are frozen.
    """

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrapped = []
    for path, linear in _named_linears(model):
        if any(skip in path for skip in skip_names):
            continue
        _set_submodule(model, path, LoRALinear(linear, rank, alpha, use_rslora))
        wrapped.append(path)
    if not wrapped:
        raise RuntimeError("no linear layers found to wrap with LoRA")
    return wrapped


def remove_lora(model: nn.Module) -> None:
    """Restore the original frozen linear layers, dropping all LoRA deltas."""

    for path, module in list(_named_linears_including_lora(model)):
        if isinstance(module, LoRALinear):
            _set_submodule(model, path, module.base)


def _named_linears_including_lora(
    module: nn.Module, prefix: str = ""
) -> list[tuple[str, nn.Module]]:
    found: list[tuple[str, nn.Module]] = []
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, LoRALinear):
            found.append((path, child))
        else:
            found.extend(_named_linears_including_lora(child, path))
    return found


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for name, p in model.named_parameters() if "lora_" in name and p.requires_grad]


def trainable_parameter_count(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
