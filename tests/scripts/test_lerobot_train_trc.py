#!/usr/bin/env python

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.train import ACPConfig
from lerobot.rl.acp_hook import build_acp_raw_batch_hook
from lerobot.rl.acp_tags import ACP_POSITIVE_TAG
from lerobot.scripts.lerobot_train_trc import validate_trc_training_config
from lerobot.scripts.lerobot_train_trc_smoke import (
    MAX_SMOKE_TRAINING_STEPS,
    format_trc_smoke_success_banner,
    validate_trc_smoke_config,
)


def _config(*, policy_type="pi05", max_delay=4, acp_enabled=True, dropout=0.0):
    return SimpleNamespace(
        policy=SimpleNamespace(type=policy_type, rtc_training_max_delay=max_delay),
        acp=ACPConfig(
            enable=acp_enabled,
            indicator_field="complementary_info.acp_indicator",
            indicator_dropout_prob=dropout,
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
        ({"dropout": 1.0}, "indicator_dropout_prob"),
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


def _smoke_config(*, steps=20, resume=False, pretrained_path=Path("pi05")):
    cfg = _config()
    cfg.steps = steps
    cfg.resume = resume
    cfg.policy.pretrained_path = pretrained_path
    cfg.policy.push_to_hub = True
    cfg.eval_freq = 100
    cfg.log_freq = 100
    cfg.wandb = SimpleNamespace(enable=True)
    return cfg


def test_trc_smoke_config_is_bounded_and_disables_external_work():
    cfg = _smoke_config()

    validate_trc_smoke_config(cfg)

    assert cfg.eval_freq == 0
    assert cfg.log_freq == 1
    assert not cfg.wandb.enable
    assert not cfg.policy.push_to_hub


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steps": 0}, "1 <= --steps"),
        ({"steps": MAX_SMOKE_TRAINING_STEPS + 1}, "1 <= --steps"),
        ({"resume": True}, "resume=true"),
        ({"pretrained_path": None}, "pretrained_path"),
    ],
)
def test_trc_smoke_config_rejects_unsafe_or_non_real_runs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate_trc_smoke_config(_smoke_config(**kwargs))


def test_trc_smoke_success_banner_contains_human_and_machine_readable_results():
    banner = format_trc_smoke_success_banner(steps=20, max_delay=37, acp_dropout=0.3)

    assert "🤗" in banner
    assert "REAL PI0.5 SMOKE TEST PASSED" in banner
    assert "[OK] Optimizer steps completed : 20" in banner
    assert "[OK] RTC training max delay D  : 37" in banner
    assert "[OK] ACP indicator dropout     : 0.300" in banner
    assert "TRC_SHORT_TRAIN: PASS (20 optimizer steps" in banner
