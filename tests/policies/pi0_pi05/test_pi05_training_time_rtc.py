#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    _apply_adarms_norm,
    _build_flow_matching_inputs,
    _reduce_training_rtc_loss,
    _sample_training_rtc_prefix_mask,
    create_sinusoidal_pos_embedding,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def test_training_rtc_uses_clean_prefix_and_per_action_time():
    actions = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    noise = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
    time = torch.tensor([0.25])
    prefix_mask = torch.tensor([[True, True, False, False]])

    x_t, model_time = _build_flow_matching_inputs(actions, noise, time, prefix_mask)

    assert model_time.tolist() == [[0.0, 0.0, 0.25, 0.25]]
    assert torch.equal(x_t[:, :2], actions[:, :2])
    assert torch.equal(x_t[:, 2:], 0.25 * noise[:, 2:] + 0.75 * actions[:, 2:])


def test_training_rtc_loss_excludes_clean_prefix():
    losses = torch.tensor([[[100.0], [100.0], [2.0], [4.0]]])
    prefix_mask = torch.tensor([[True, True, False, False]])

    mean_loss = _reduce_training_rtc_loss(losses, prefix_mask, reduction="mean")
    per_sample_loss = _reduce_training_rtc_loss(losses, prefix_mask, reduction="none")

    assert mean_loss.item() == pytest.approx(3.0)
    assert per_sample_loss.tolist() == pytest.approx([3.0])


def test_training_rtc_samples_valid_prefixes():
    torch.manual_seed(7)
    mask = _sample_training_rtc_prefix_mask(
        batch_size=32,
        action_horizon=10,
        max_delay=4,
        device=torch.device("cpu"),
    )

    assert mask is not None
    assert mask.shape == (32, 10)
    lengths = mask.sum(dim=1)
    assert torch.all(lengths <= 4)
    assert torch.all(mask[:, 1:] <= mask[:, :-1])


def test_per_action_time_embedding_preserves_token_axis():
    embedding = create_sinusoidal_pos_embedding(
        torch.tensor([[0.0, 0.5, 1.0]]),
        dimension=8,
        min_period=0.004,
        max_period=4.0,
        device=torch.device("cpu"),
    )

    assert embedding.shape == (1, 3, 8)


class _LegacyAdaRMS(nn.Module):
    """Minimal form of the custom transformers AdaRMS used by this checkout."""

    def __init__(self, width: int):
        super().__init__()
        self.cond_dim = width
        self.eps = 1e-6
        self.dense = nn.Linear(width, width * 3)

    def _norm(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x, cond=None):
        normed = self._norm(x)
        if cond is None:
            return normed.to(x.dtype), None
        modulation = self.dense(cond)
        if x.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        return (normed * (1 + scale.float()) + shift.float()).to(x.dtype), gate.to(x.dtype)


def test_per_action_adarms_matches_uniform_scalar_condition():
    torch.manual_seed(0)
    batch, tokens, width = 2, 4, 8
    norm = _LegacyAdaRMS(width)
    x = torch.randn(batch, tokens, width)
    scalar_cond = torch.randn(batch, width)
    token_cond = scalar_cond[:, None, :].expand(batch, tokens, width)

    scalar_out, scalar_gate = _apply_adarms_norm(norm, x, scalar_cond)
    token_out, token_gate = _apply_adarms_norm(norm, x, token_cond)

    torch.testing.assert_close(token_out, scalar_out)
    torch.testing.assert_close(token_gate, scalar_gate.expand_as(token_gate))


class _FakePI05Core(nn.Module):
    def __init__(self, losses: torch.Tensor):
        super().__init__()
        self.losses = losses
        self.prefix_mask = None

    def forward(self, images, img_masks, tokens, masks, actions, prefix_mask=None):
        del images, img_masks, tokens, masks, actions
        self.prefix_mask = prefix_mask
        return self.losses


def test_pi05_policy_forward_passes_prefix_mask_and_reduces_postfix(monkeypatch):
    prefix_mask = torch.tensor([[True, True, False, False]])
    monkeypatch.setattr(
        "lerobot.policies.pi05.modeling_pi05._sample_training_rtc_prefix_mask",
        lambda **_: prefix_mask,
    )

    policy = object.__new__(PI05Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        max_action_dim=1,
        rtc_training_max_delay=2,
        output_features={ACTION: SimpleNamespace(shape=(1,))},
    )
    policy.model = _FakePI05Core(torch.tensor([[[100.0], [100.0], [2.0], [4.0]]]))
    policy._preprocess_images = lambda _: ([], [])
    batch = {
        ACTION: torch.zeros(1, 4, 1),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.long),
    }

    loss, loss_dict = policy.forward(batch)

    assert torch.equal(policy.model.prefix_mask, prefix_mask)
    assert loss.item() == pytest.approx(3.0)
    assert loss_dict["rtc_prefix_length"] == pytest.approx(2.0)


@pytest.mark.parametrize("max_delay", [-1, 5])
def test_pi05_config_rejects_invalid_training_rtc_delay(max_delay):
    with pytest.raises(ValueError, match="rtc_training_max_delay"):
        PI05Config(chunk_size=5, n_action_steps=5, rtc_training_max_delay=max_delay)
