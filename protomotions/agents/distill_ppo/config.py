# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from dataclasses import dataclass, field
from typing import Optional

from protomotions.agents.ppo.config import PPOAgentConfig


@dataclass
class PPOLossScheduleConfig:
    enabled: bool = field(
        default=False,
        metadata={"help": "Enable scheduling of the PPO surrogate loss weight."},
    )
    init_coef: float = field(
        default=1.0,
        metadata={"help": "Initial PPO loss coefficient.", "min": 0.0},
    )
    end_coef: float = field(
        default=1.0,
        metadata={"help": "Final PPO loss coefficient.", "min": 0.0},
    )
    start_epoch: int = field(
        default=0,
        metadata={"help": "Epoch to start ramping the PPO loss coefficient.", "min": 0},
    )
    end_epoch: int = field(
        default=0,
        metadata={"help": "Epoch to finish ramping the PPO loss coefficient.", "min": 0},
    )


@dataclass
class DistillPPOAgentConfig(PPOAgentConfig):
    _target_: str = "protomotions.agents.distill_ppo.agent.DistillPPO"

    expert_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a frozen expert checkpoint used for action distillation."},
    )
    action_loss_coef: float = field(
        default=1.0,
        metadata={"help": "Weight applied to the expert action matching loss."},
    )
    ppo_loss_schedule: PPOLossScheduleConfig = field(
        default_factory=PPOLossScheduleConfig,
        metadata={"help": "Schedule for scaling the PPO surrogate loss term."},
    )
