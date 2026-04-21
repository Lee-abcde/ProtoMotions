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
class ActionLossScheduleConfig:
    enabled: bool = field(
        default=False,
        metadata={"help": "Enable scheduling of the distillation action loss weight."},
    )
    init_coef: float = field(
        default=1.0,
        metadata={"help": "Initial action loss coefficient.", "min": 0.0},
    )
    end_coef: float = field(
        default=1.0,
        metadata={"help": "Final action loss coefficient.", "min": 0.0},
    )
    start_epoch: int = field(
        default=0,
        metadata={"help": "Epoch to start ramping the action loss coefficient.", "min": 0},
    )
    end_epoch: int = field(
        default=0,
        metadata={"help": "Epoch to finish ramping the action loss coefficient.", "min": 0},
    )


@dataclass
class MiniEpochScheduleConfig:
    enabled: bool = field(
        default=False,
        metadata={"help": "Enable scheduling of the number of mini-epochs per update."},
    )
    init_num_mini_epochs: int = field(
        default=1,
        metadata={"help": "Initial number of mini-epochs per update.", "min": 1},
    )
    end_num_mini_epochs: int = field(
        default=1,
        metadata={"help": "Final number of mini-epochs per update.", "min": 1},
    )
    start_epoch: int = field(
        default=0,
        metadata={"help": "Epoch to start transitioning mini-epochs.", "min": 0},
    )
    end_epoch: int = field(
        default=0,
        metadata={"help": "Epoch to finish transitioning mini-epochs.", "min": 0},
    )


@dataclass
class ActorLRScheduleConfig:
    enabled: bool = field(
        default=False,
        metadata={"help": "Enable scheduling of the actor optimizer learning rate."},
    )
    init_lr: float = field(
        default=2e-5,
        metadata={"help": "Initial actor learning rate.", "min": 0.0},
    )
    end_lr: float = field(
        default=2e-5,
        metadata={"help": "Final actor learning rate.", "min": 0.0},
    )
    start_epoch: int = field(
        default=0,
        metadata={"help": "Epoch to start transitioning the actor learning rate.", "min": 0},
    )
    end_epoch: int = field(
        default=0,
        metadata={"help": "Epoch to finish transitioning the actor learning rate.", "min": 0},
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
    action_loss_schedule: ActionLossScheduleConfig = field(
        default_factory=ActionLossScheduleConfig,
        metadata={"help": "Schedule for scaling the distillation action loss term."},
    )
    ppo_loss_schedule: PPOLossScheduleConfig = field(
        default_factory=PPOLossScheduleConfig,
        metadata={"help": "Schedule for scaling the PPO surrogate loss term."},
    )
    mini_epoch_schedule: MiniEpochScheduleConfig = field(
        default_factory=MiniEpochScheduleConfig,
        metadata={"help": "Schedule for changing mini-epochs per update over training."},
    )
    actor_lr_schedule: ActorLRScheduleConfig = field(
        default_factory=ActorLRScheduleConfig,
        metadata={"help": "Schedule for the actor optimizer learning rate."},
    )
