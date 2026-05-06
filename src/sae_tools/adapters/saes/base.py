from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


class SAEAdapter(Protocol):
    cfg: Any
    W_dec: torch.Tensor

    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class TopKSAEConfig:
    source: str
    hook_layer: int
    hook_name: str
    d_in: int
    d_sae: int
    top_k: int
    device: str
    dtype: str
    apply_b_dec_to_input: bool = True
