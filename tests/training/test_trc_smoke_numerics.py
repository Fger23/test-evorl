#!/usr/bin/env python

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.scripts.lerobot_train import update_policy


class _FakeAccelerator:
    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=False)


class _ScalarLossPolicy(nn.Module):
    def __init__(self, multiplier: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.multiplier = multiplier

    def forward(self, batch):
        return self.weight * self.multiplier, {}


def _update(multiplier: float):
    policy = _ScalarLossPolicy(multiplier)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    metrics = SimpleNamespace()
    return update_policy(
        metrics,
        policy,
        batch={},
        optimizer=optimizer,
        grad_clip_norm=1.0,
        accelerator=_FakeAccelerator(),
        fail_on_nonfinite=True,
    )


@pytest.mark.parametrize("multiplier", [float("nan"), float("inf")])
def test_trc_smoke_rejects_nonfinite_loss(multiplier):
    with pytest.raises(FloatingPointError, match="loss is non-finite"):
        _update(multiplier)


def test_trc_smoke_rejects_zero_gradient():
    with pytest.raises(RuntimeError, match="gradient norm is zero"):
        _update(0.0)
