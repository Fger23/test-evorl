#!/usr/bin/env python

from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.train import ACPConfig
from lerobot.rl.acp_hook import build_acp_raw_batch_hook
from lerobot.rl.acp_tags import ACP_POSITIVE_TAG
from lerobot.scripts.lerobot_train_trc import validate_trc_training_config


def _config(*, policy_type="pi05", max_delay=4, acp_enabled=True):
    return SimpleNamespace(
        policy=SimpleNamespace(type=policy_type, rtc_training_max_delay=max_delay),
        acp=ACPConfig(
            enable=acp_enabled,
            indicator_field="complementary_info.acp_indicator",
            indicator_dropout_prob=0.0,
        ),
    )


def test_trc_entry_accepts_pi05_training_rtc_with_acp():
    validate_trc_training_config(_config())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"policy_type": "pi0"}, "only supports"),
        ({"max_delay": 0}, "rtc_training_max_delay"),
        ({"acp_enabled": False}, "acp.enable"),
    ],
)
def test_trc_entry_rejects_missing_conditioning(kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate_trc_training_config(_config(**kwargs))


def test_trc_entry_keeps_acp_positive_prompt_conditioning():
    cfg = _config()
    validate_trc_training_config(cfg)
    hook = build_acp_raw_batch_hook(cfg.acp, seed=0)
    batch = {
        "task": ["pick bottle"],
        "complementary_info.acp_indicator": torch.tensor([1], dtype=torch.int64),
    }

    conditioned = hook(batch, 0)

    assert conditioned["task"] == [f"pick bottle\n{ACP_POSITIVE_TAG}"]
