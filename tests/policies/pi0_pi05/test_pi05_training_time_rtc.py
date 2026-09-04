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
    PaliGemmaWithExpertModel,
    _apply_adarms_norm,
    _build_flow_matching_inputs,
    _forward_expert_suffix_with_per_token_adarms,
    _integrate_training_rtc_actions,
    _prepare_training_rtc_inference_prefix,
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


def test_training_rtc_inference_pads_and_hard_clamps_clean_prefix():
    noise = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    raw_prefix = torch.tensor([[[-1.0, -2.0], [-3.0, -4.0], [99.0, 99.0]]])
    clean_prefix, prefix_mask = _prepare_training_rtc_inference_prefix(
        noise=noise,
        action_prefix=raw_prefix,
        inference_delay=2,
        max_delay=3,
    )

    assert prefix_mask.tolist() == [[True, True, False, False]]
    torch.testing.assert_close(clean_prefix[0, :2, :2], raw_prefix[0, :2])
    assert torch.count_nonzero(clean_prefix[0, :2, 2:]) == 0

    calls = []

    def denoise(x_t, model_time):
        calls.append((x_t.clone(), model_time.clone()))
        return torch.ones_like(x_t)

    result = _integrate_training_rtc_actions(
        denoise_fn=denoise,
        noise=noise,
        clean_prefix=clean_prefix,
        prefix_mask=prefix_mask,
        num_steps=2,
    )

    torch.testing.assert_close(result[:, :2], clean_prefix[:, :2])
    torch.testing.assert_close(result[:, 2:], noise[:, 2:] - 1.0)
    assert len(calls) == 2
    for x_t, model_time in calls:
        torch.testing.assert_close(x_t[:, :2], clean_prefix[:, :2])
        assert model_time[0, :2].tolist() == [0.0, 0.0]
    assert calls[0][1][0, 2:].tolist() == [1.0, 1.0]
    assert calls[1][1][0, 2:].tolist() == [0.5, 0.5]


def test_training_rtc_inference_delay_zero_uses_per_token_time_without_prefix():
    noise = torch.zeros(1, 3, 2)
    clean_prefix, prefix_mask = _prepare_training_rtc_inference_prefix(
        noise=noise,
        action_prefix=None,
        inference_delay=0,
        max_delay=2,
    )
    seen_time = []

    result = _integrate_training_rtc_actions(
        denoise_fn=lambda x_t, model_time: seen_time.append(model_time.clone()) or torch.zeros_like(x_t),
        noise=noise,
        clean_prefix=clean_prefix,
        prefix_mask=prefix_mask,
        num_steps=1,
    )

    assert not prefix_mask.any()
    assert seen_time[0].shape == (1, 3)
    torch.testing.assert_close(result, noise)


@pytest.mark.parametrize(
    ("delay", "max_delay", "prefix", "match"),
    [
        (3, 2, torch.zeros(1, 3, 1), "trained maximum"),
        (4, 4, torch.zeros(1, 4, 1), "smaller than chunk_size"),
        (2, 3, torch.zeros(1, 1, 1), "contains 1 steps"),
        (1, 3, torch.tensor([[[float("nan")]]]), "NaN or Inf"),
    ],
)
def test_training_rtc_inference_rejects_invalid_prefix(delay, max_delay, prefix, match):
    with pytest.raises(ValueError, match=match):
        _prepare_training_rtc_inference_prefix(
            noise=torch.zeros(1, 4, 2),
            action_prefix=prefix,
            inference_delay=delay,
            max_delay=max_delay,
        )


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


def test_cached_expert_suffix_supports_per_action_adarms_without_mutating_prefix_cache():
    """Exercise the real Transformers suffix path used by PI0.5 denoising."""
    from transformers import GemmaConfig
    from transformers.models.gemma import modeling_gemma

    torch.manual_seed(0)
    batch_size, prefix_steps, action_steps, width = 2, 3, 5, 8
    common_config = {
        "head_dim": 4,
        "hidden_size": width,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "vocab_size": 32,
    }
    prefix_config = GemmaConfig(**common_config, use_adarms=False)
    expert_config = GemmaConfig(**common_config, use_adarms=True, adarms_cond_dim=width)
    prefix_config._attn_implementation = "eager"  # noqa: SLF001
    expert_config._attn_implementation = "eager"  # noqa: SLF001
    prefix_model = modeling_gemma.GemmaModel(prefix_config).eval()
    expert_model = modeling_gemma.GemmaModel(expert_config).eval()

    prefix_embeds = torch.randn(batch_size, prefix_steps, width)
    suffix_embeds = torch.randn(batch_size, action_steps, width)
    position_ids = torch.arange(prefix_steps, prefix_steps + action_steps).expand(batch_size, -1)
    attention_mask = torch.zeros(batch_size, 1, action_steps, prefix_steps + action_steps)
    attention_mask[:, :, :, 0] = torch.finfo(attention_mask.dtype).min

    with torch.no_grad():
        prefix_cache = prefix_model(inputs_embeds=prefix_embeds, use_cache=True).past_key_values
    cache_before = [
        (prefix_cache[layer][0].clone(), prefix_cache[layer][1].clone())
        for layer in range(len(prefix_cache))
    ]

    wrapper = object.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(wrapper)
    wrapper.gemma_expert = SimpleNamespace(model=expert_model)
    scalar_cond = torch.randn(batch_size, width)
    token_cond = scalar_cond[:, None, :].expand(batch_size, action_steps, width)

    with torch.no_grad():
        scalar_output = wrapper.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=prefix_cache,
            inputs_embeds=[None, suffix_embeds],
            use_cache=False,
            adarms_cond=[None, scalar_cond],
        )[0][1]
        token_output = wrapper.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=prefix_cache,
            inputs_embeds=[None, suffix_embeds],
            use_cache=False,
            adarms_cond=[None, token_cond],
        )[0][1]
        nonuniform_output = _forward_expert_suffix_with_per_token_adarms(
            model=expert_model,
            inputs_embeds=suffix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=prefix_cache,
            adarms_cond=token_cond.clone().index_add(
                1,
                torch.tensor([0, 1]),
                torch.ones(batch_size, 2, width),
            ),
        )

    assert scalar_output.shape == token_output.shape == nonuniform_output.shape == (
        batch_size,
        action_steps,
        width,
    )
    assert torch.isfinite(nonuniform_output).all()
    torch.testing.assert_close(token_output, scalar_output)
    assert prefix_cache.get_seq_length() == prefix_steps
    for layer, (key_before, value_before) in enumerate(cache_before):
        torch.testing.assert_close(prefix_cache[layer][0], key_before)
        torch.testing.assert_close(prefix_cache[layer][1], value_before)


def _initialize_empty_pi05_policy(policy, config, **_kwargs):
    nn.Module.__init__(policy)
    policy.config = config


def test_pi05_from_pretrained_fails_closed_when_weights_cannot_be_read(monkeypatch):
    import transformers.utils

    monkeypatch.setattr(PI05Policy, "__init__", _initialize_empty_pi05_policy)

    def fail_cached_file(*_args, **_kwargs):
        raise OSError("missing or corrupt checkpoint")

    monkeypatch.setattr(transformers.utils, "cached_file", fail_cached_file)

    with pytest.raises(RuntimeError, match="Failed to load PI0.5 checkpoint weights"):
        PI05Policy.from_pretrained(
            "/tmp/not-a-policy",
            config=SimpleNamespace(device="cpu"),
        )


def test_pi05_from_pretrained_fails_closed_on_strict_state_dict_mismatch(monkeypatch):
    import safetensors.torch
    import transformers.utils

    monkeypatch.setattr(PI05Policy, "__init__", _initialize_empty_pi05_policy)
    monkeypatch.setattr(transformers.utils, "cached_file", lambda *_args, **_kwargs: "/tmp/model")
    monkeypatch.setattr(
        safetensors.torch,
        "load_file",
        lambda *_args, **_kwargs: {"unexpected.weight": torch.ones(1)},
    )

    with pytest.raises(RuntimeError, match="Failed to load PI0.5 checkpoint weights"):
        PI05Policy.from_pretrained(
            "/tmp/mismatched-policy",
            config=SimpleNamespace(device="cpu"),
            strict=True,
        )


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
