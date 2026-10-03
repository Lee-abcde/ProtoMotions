# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage two: one sparse conditional prior for the frozen joint PVQ motor policy.

Tracker environments and native posterior observations come from the stage-one
source manifest. Only the prior's extra conditions and architecture live here.
"""

from protomotions.agents.joint_prior.config import JointPriorConfig


def prior_config(num_bodies: int) -> JointPriorConfig:
    return JointPriorConfig(num_bodies=num_bodies)
