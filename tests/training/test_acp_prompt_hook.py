#!/usr/bin/env python

import pytest
import torch
from torch.utils.data import DataLoader

from lerobot.configs.train import ACPConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.acp_hook import build_acp_raw_batch_hook
from lerobot.rl.acp_tags import ACP_NEGATIVE_TAG, ACP_POSITIVE_TAG


def test_acp_hook_accepts_integer_indicators():
    hook = build_acp_raw_batch_hook(
        ACPConfig(
            enable=True,
            indicator_field="acp_indicator",
            indicator_dropout_prob=0.0,
        ),
        seed=42,
    )
    batch = {
        "task": ["pick bottle", "place bottle"],
        "acp_indicator": torch.tensor([1, 0], dtype=torch.int64),
    }

    out = hook(batch, 0)
    assert out["task"] == [
        f"pick bottle\n{ACP_POSITIVE_TAG}",
        f"place bottle\n{ACP_NEGATIVE_TAG}",
    ]


def test_acp_hook_injects_tags():
    hook = build_acp_raw_batch_hook(
        ACPConfig(
            enable=True,
            indicator_field="complementary_info.acp_indicator",
            indicator_dropout_prob=0.0,
        ),
        seed=42,
    )
    batch = {
        "task": ["pick bottle", "place bottle"],
        "complementary_info.acp_indicator": torch.tensor([1, 0], dtype=torch.int64),
    }

    out = hook(batch, 0)
    assert out["task"] == [f"pick bottle\n{ACP_POSITIVE_TAG}", f"place bottle\n{ACP_NEGATIVE_TAG}"]


def test_acp_hook_dropout_keeps_original_task():
    hook = build_acp_raw_batch_hook(
        ACPConfig(
            enable=True,
            indicator_field="complementary_info.acp_indicator",
            indicator_dropout_prob=1.0,
        ),
        seed=42,
    )
    batch = {
        "task": ["pick bottle", "place bottle"],
        "complementary_info.acp_indicator": torch.tensor([1, 0], dtype=torch.int64),
    }

    out = hook(batch, 0)
    assert out["task"] == ["pick bottle", "place bottle"]


def test_acp_hook_dropout_is_identical_after_resume_at_same_step():
    cfg = ACPConfig(
        enable=True,
        indicator_field="complementary_info.acp_indicator",
        indicator_dropout_prob=0.5,
    )
    batch = {
        "task": [f"task {i}" for i in range(32)],
        "index": torch.arange(100, 132),
        "complementary_info.acp_indicator": torch.tensor([i % 2 for i in range(32)]),
    }

    uninterrupted_hook = build_acp_raw_batch_hook(cfg, seed=42)
    resumed_hook = build_acp_raw_batch_hook(cfg, seed=42)
    uninterrupted = uninterrupted_hook({key: value.copy() if isinstance(value, list) else value for key, value in batch.items()}, 17)
    resumed = resumed_hook({key: value.copy() if isinstance(value, list) else value for key, value in batch.items()}, 17)

    assert uninterrupted["task"] == resumed["task"]
    assert any("Advantage:" in task for task in uninterrupted["task"])
    assert any("Advantage:" not in task for task in uninterrupted["task"])


def test_acp_hook_missing_indicator_skips():
    hook = build_acp_raw_batch_hook(
        ACPConfig(enable=True, indicator_field="missing_field"),
        seed=42,
    )
    batch = {"task": ["pick bottle"]}
    with pytest.raises(KeyError, match="missing_field"):
        hook(batch, 0)


def test_acp_hook_non_integer_type_raises():
    hook = build_acp_raw_batch_hook(
        ACPConfig(
            enable=True,
            indicator_field="acp_indicator",
            indicator_dropout_prob=0.0,
        ),
        seed=42,
    )
    batch = {
        "task": ["pick bottle", "place bottle"],
        "acp_indicator": torch.tensor([1.0, 0.0], dtype=torch.float32),
    }
    with pytest.raises(TypeError, match="integer 0/1"):
        hook(batch, 0)


def test_acp_hook_non_binary_value_raises():
    hook = build_acp_raw_batch_hook(
        ACPConfig(
            enable=True,
            indicator_field="acp_indicator",
            indicator_dropout_prob=0.0,
        ),
        seed=42,
    )
    batch = {
        "task": ["pick bottle", "place bottle"],
        "acp_indicator": torch.tensor([2, 0], dtype=torch.int64),
    }
    with pytest.raises(ValueError, match="must be 0 or 1"):
        hook(batch, 0)


def _load_real_batch_or_skip(batch_size: int) -> dict:
    repo_id = "local/eval_so101_pick_place_bottle_v0"
    try:
        dataset = LeRobotDataset(repo_id, download_videos=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return next(iter(dataloader))
    except Exception as e:
        pytest.skip(f"Cannot load local dataset '{repo_id}': {e}")


@pytest.mark.parametrize("batch_size", [1, 3])
def test_acp_hook_with_real_local_dataset_batch(batch_size: int):
    batch = _load_real_batch_or_skip(batch_size)
    assert "task" in batch
    assert "complementary_info.acp_indicator" not in batch

    indicators = torch.randint(0, 2, (batch_size,), dtype=torch.int64)
    batch["complementary_info.acp_indicator"] = indicators

    hook = build_acp_raw_batch_hook(
        ACPConfig(
            enable=True,
            indicator_field="complementary_info.acp_indicator",
            indicator_dropout_prob=0.0,
        ),
        seed=42,
    )
    out = hook(batch, 0)

    assert len(out["task"]) == batch_size
    for i in range(batch_size):
        expected_tag = ACP_POSITIVE_TAG if int(indicators[i].item()) == 1 else ACP_NEGATIVE_TAG
        assert out["task"][i].endswith(expected_tag)
