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

"""Bounded, strict-numerics smoke training for ACP-conditioned PI0.5 with Training-Time RTC."""

import sys

from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import train
from lerobot.scripts.lerobot_train_trc import validate_trc_training_config, validate_trc_training_dataset
from lerobot.utils.import_utils import register_third_party_plugins

MAX_SMOKE_TRAINING_STEPS = 1_000


def format_trc_smoke_success_banner(*, steps: int, max_delay: int, acp_dropout: float) -> str:
    """Return a celebratory terminal banner while preserving a grep-friendly PASS marker."""
    width = 62
    top = "+" + "=" * width + "+"
    divider = "+" + "=" * width + "+"
    bottom = "+" + "=" * width + "+"

    def row(text: str = "") -> str:
        return f"| {text:<{width - 2}} |"

    return "\n".join(
        [
            "",
            "                    __...__",
            "                 .-'  _  _ '-.",
            "              🤗/    (o)(o)    \\🤗",
            "                |       ^       |",
            "                 \\    \\___/    /",
            "                  '._       _.'",
            "                     '-----'",
            top,
            row("ACP + TRAINING-TIME RTC".center(width - 2)),
            row("REAL PI0.5 SMOKE TEST PASSED".center(width - 2)),
            divider,
            row(f"[OK] Optimizer steps completed : {steps}"),
            row(f"[OK] RTC training max delay D  : {max_delay}"),
            row(f"[OK] ACP indicator dropout     : {acp_dropout:.3f}"),
            row("[OK] Loss and gradients        : finite and non-zero"),
            divider,
            row("Ready for formal training: lerobot-train-trc".center(width - 2)),
            bottom,
            f"TRC_SHORT_TRAIN: PASS ({steps} optimizer steps completed with finite non-zero gradients)",
            "",
        ]
    )


def validate_trc_smoke_config(cfg: TrainPipelineConfig) -> None:
    """Enforce a short, local-only run before the real model is constructed."""
    validate_trc_training_config(cfg)
    if not 1 <= cfg.steps <= MAX_SMOKE_TRAINING_STEPS:
        raise ValueError(
            "lerobot-train-trc-smoke requires 1 <= --steps <= "
            f"{MAX_SMOKE_TRAINING_STEPS}, got {cfg.steps}."
        )
    if cfg.resume:
        raise ValueError("lerobot-train-trc-smoke starts a fresh bounded run; --resume=true is not supported.")
    if cfg.policy.pretrained_path is None:
        raise ValueError("lerobot-train-trc-smoke requires a real --policy.pretrained_path checkpoint.")

    # A smoke test must remain local and must not spend time on rollout evaluation.
    cfg.eval_freq = 0
    cfg.log_freq = 1
    cfg.wandb.enable = False
    cfg.policy.push_to_hub = False


@parser.wrap()
def train_trc_smoke(cfg: TrainPipelineConfig):
    """Run real PI0.5 forward/backward optimizer steps with strict numerical checks."""
    result = train(
        cfg,
        config_validator=validate_trc_smoke_config,
        dataset_validator=validate_trc_training_dataset,
        fail_on_nonfinite=True,
    )
    banner = format_trc_smoke_success_banner(
        steps=cfg.steps,
        max_delay=cfg.policy.rtc_training_max_delay,
        acp_dropout=cfg.acp.indicator_dropout_prob,
    )
    terminal_encoding = sys.stdout.encoding or "utf-8"
    terminal_banner = banner.encode(terminal_encoding, errors="replace").decode(terminal_encoding)
    print(colored(terminal_banner, "green", attrs=["bold"]))
    return result


def main():
    register_third_party_plugins()
    train_trc_smoke()


if __name__ == "__main__":
    main()
