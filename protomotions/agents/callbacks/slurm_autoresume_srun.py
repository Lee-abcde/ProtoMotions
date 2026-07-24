# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time

import wandb

from pytorch_lightning import Callback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from protomotions.agents.ppo import PPO
else:
    PPO = object

log = logging.getLogger(__name__)


def wandb_run_exists():
    return isinstance(wandb.run, wandb.sdk.wandb_run.Run)


class AutoResumeCallbackSrun(Callback):
    def __init__(self, autoresume_after=12600) -> None:
        self.start_time = None
        self.autoresume_after = autoresume_after

        print("************************************")
        print("will autoresume after ", self.autoresume_after)

    def _check_autoresume(self, agent: PPO):
        agent.fabric.strategy.barrier()

        if self.start_time is None:
            self.start_time = time.time()

        # Rank-local clocks can straddle the threshold even after the barrier.
        # If only one rank enters save(), its collectives diverge from ranks that
        # continue training. Let rank 0 decide and broadcast that decision.
        should_autoresume = False
        if agent.fabric.global_rank == 0:
            should_autoresume = (
                time.time() - self.start_time >= self.autoresume_after
            )
        should_autoresume = agent.fabric.broadcast(should_autoresume, src=0)

        if should_autoresume:
            log.info("Should autoresume!")

            agent.save()

            agent._should_stop = True
            log.info("Autoresume checkpoint saved; stopping after this epoch.")

    def before_play_steps(self, agent: PPO) -> None:
        self._check_autoresume(agent)

    def on_fit_start(self, agent: PPO) -> None:
        pass

    def on_fit_end(self, agent: PPO) -> None:
        pass
