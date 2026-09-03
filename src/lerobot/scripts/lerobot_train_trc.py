#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ACP-conditioned PI05 training with Training-Time Real-Time Chunking."""

import logging
from typing import Any

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.rl.acp_dataset_stats import validate_acp_training_dataset
from lerobot.scripts.lerobot_train import train
from lerobot.utils.import_utils import register_third_party_plugins


def validate_trc_training_config(cfg: TrainPipelineConfig) -> None:
    """Require the two conditioning paths promised by the TRC entrypoint."""
    if cfg.policy is None or cfg.policy.type != "pi05":
        policy_type = None if cfg.policy is None else cfg.policy.type
        raise ValueError(f"lerobot-train-trc only supports --policy.type=pi05, got {policy_type!r}.")

    max_delay = int(getattr(cfg.policy, "rtc_training_max_delay", 0))
    if max_delay <= 0:
        raise ValueError(
            "lerobot-train-trc requires --policy.rtc_training_max_delay=<positive controller steps>."
        )
    if not cfg.acp.enable:
        raise ValueError(
            "lerobot-train-trc requires --acp.enable=true so ACP prompt conditioning is preserved."
        )
    if cfg.acp.indicator_dropout_prob >= 1.0:
        raise ValueError(
            "lerobot-train-trc requires --acp.indicator_dropout_prob<1 so at least some Advantage prompts remain."
        )

    logging.info(
        "TRC training enabled: policy=pi05 rtc_training_max_delay=%d acp_indicator_field=%s "
        "acp_indicator_dropout_prob=%.3f",
        max_delay,
        cfg.acp.indicator_field,
        cfg.acp.indicator_dropout_prob,
    )


def validate_trc_training_dataset(dataset: Any, cfg: TrainPipelineConfig) -> None:
    """Fail before policy creation when the ACP annotations cannot train both prompts."""
    validate_acp_training_dataset(dataset, cfg.acp.indicator_field, require_both_classes=True)


@parser.wrap()
def train_trc(cfg: TrainPipelineConfig):
    """Run the shared trainer after enforcing ACP plus Training-Time RTC."""
    return train(
        cfg,
        config_validator=validate_trc_training_config,
        dataset_validator=validate_trc_training_dataset,
    )


def main():
    register_third_party_plugins()
    train_trc()


if __name__ == "__main__":
    main()
